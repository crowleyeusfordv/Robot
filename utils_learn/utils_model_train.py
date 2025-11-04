# ⚡ FK, IK, Sequence Learning 训练流程
import time, numpy as np, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
import pandas as pd
from utils_learn.flexible_networks import FlexibleMLP
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F


# ---------- 工具函数 ----------
def select_cols(data_df, names):
    return data_df[names].values


def split_data(data_df, in_cols, out_cols):
    X = select_cols(data_df, in_cols)
    Y = select_cols(data_df, out_cols)
    from sklearn.model_selection import train_test_split
    return train_test_split(X, Y, test_size=0.2, random_state=42)


def compute_fk_loss(y_pred, y_true, w_pos=0.9, w_ori=0.1):
    """
    FK 损失
    参数
    ----
    y_pred : [B, 12]  预测位姿（归一化）
    y_true : [B, 12]  真值位姿（仅监控）
    w_pos  : 位置误差权重
    w_ori  : 姿态误差权重
    返回
    ----
    loss : 标量张量
    info : dict
    """
    # 1. 位置
    loss_pos = F.mse_loss(y_pred[:, :3], y_true[:, :3])

    # 2. 旋转矩阵误差（Frobenius 距离）
    R_pred = y_pred[:, 3:]  # [B, 9]
    R_true = y_true[:, 3:]  # [B, 9]
    loss_ori = F.mse_loss(R_pred, R_true)
    # 3. 加权
    # loss = w_pos * loss_pos + w_ori * loss_ori
    # 4. 监控
    with torch.no_grad():
        mae = torch.mean(torch.abs(y_pred - y_true)).item()
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
        pos_error = torch.norm(y_pred[:, :3] - y_true[:, :3], dim=1).mean().item()
        ori_error = torch.norm(R_pred - R_true, dim=1).mean().item()

    loss = F.mse_loss(y_pred, y_true)
    return loss, {'mae': mae, 'rmse': rmse, 'position_error': pos_error, 'orientation_error': ori_error}


def compute_ik_loss(q_pred, q_true,
                    pose_true=None, fk_ref=None,
                    w_pos=0.9, w_ori=0.1):
    """
    IK 损失
    参数
    ----
    q_pred : [B, 5]  预测关节角（归一化）
    q_true : [B, 5]  真值关节角（仅监控）
    mat_true:[B, 12] 真值末端矩阵 [x,y,z | 9-elements-of-R]
    fk_ref : 冻结的 FK 网络，输入 q 输出 [B,12]
    w_pos  : 位置误差权重
    w_ori  : 姿态误差权重

    返回
    ----
    loss : 标量张量
    info : dict
    """
    # ---- 1. 关节角监控（无梯度） ----
    with torch.no_grad():
        joint_mae = torch.mean(torch.abs(q_pred - q_true)).item()
        joint_rmse = torch.sqrt(torch.mean((q_pred - q_true) ** 2)).item()

    # ---- 2. 无矩阵监督 → 退化为关节 MSE ----
    if pose_true is None or fk_ref is None:
        loss = F.mse_loss(q_pred, q_true)
        info = {'joint_mae': joint_mae, 'joint_rmse': joint_rmse}
        return loss, info

    # ---- 3. FK 监督 → 末端矩阵损失 ----
    pred_mat = fk_ref(q_pred)  # [B,12]

    # 3.1 位置损失
    loss_pos = F.mse_loss(pred_mat[:, :3], pose_true[:, :3])

    # 3.2 旋转矩阵损失（Frobenius）
    loss_ori = F.mse_loss(pred_mat[:, 3:], pose_true[:, 3:])
    # # 3.3 加权总损失
    # loss = w_pos * loss_pos + w_ori * loss_ori

    loss = F.mse_loss(pred_mat, pose_true)

    # ---- 4. 监控指标 ----
    with torch.no_grad():
        pos_err = torch.norm(pred_mat[:, :3] - pose_true[:, :3], dim=1).mean().item()
        ori_err = torch.norm(pred_mat[:, 3:] - pose_true[:, 3:], dim=1).mean().item()

    info = {'joint_mae': joint_mae,
            'joint_rmse': joint_rmse,
            'position_error': pos_err,
            'orientation_error': ori_err}

    return loss, info


def plot_training_curves(history: dict, save_path: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(history['train'], label=f"Train {history['metric']}")
    plt.plot(history['test'], label=f"Test  {history['metric']}")
    plt.xlabel('Epoch');
    plt.ylabel(history['metric']);
    plt.title(f"{history['metric']} Curve")
    plt.legend();
    plt.tight_layout();
    plt.savefig(save_path, dpi=300);
    plt.close()
    print(f'📈 曲线已保存 → {save_path}')


# ---------- 唯一入口 ----------
def train_dofbot_model(data_path,
                       model_type='mlp',  # 'mlp' | 'mdn' | 'lstm'
                       mode='fk',  # 'fk'  | 'ik'
                       in_cols=None,  # list[str] 仅 fk 有效
                       out_cols=None,  # list[str] 仅 ik 有效
                       epochs=1000,
                       lr=1e-3,
                       min_lr=1e-3,
                       num_mixtures=5,
                       hidden_layers=[100, 30],
                       seq_len=10,
                       fk_path=None,
                       fk_hidden_layers=None,
                       w_pos=0.9,
                       w_ori=0.1
                       ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # 0. 确定输入/输出列名
    data_df = pd.read_csv(data_path)
    if mode == 'fk':
        in_cols = in_cols or ['q1_sin', 'q1_cos', 'q2_sin', 'q2_cos', 'q3_sin', 'q3_cos', 'q4_sin', 'q4_cos', 'q5_sin',
                              'q5_cos']
        out_cols = out_cols or ['x', 'y', 'z', 'nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az']  # 默认 xyz+orn
    else:  # ik
        in_cols = in_cols or ['x', 'y', 'z', 'nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'ax', 'ay', 'az']
        out_cols = out_cols or ['q1_sin', 'q1_cos', 'q2_sin', 'q2_cos', 'q3_sin', 'q3_cos', 'q4_sin', 'q4_cos',
                                'q5_sin', 'q5_cos']

        # 1. 加载冻结 FK
        fk_ref = FlexibleMLP(len(out_cols), len(in_cols), hidden_layers=fk_hidden_layers, dropout=0.0,
                             activation='ReLU', block_type='res',
                             num_blocks=1).to(device)
        fk_ref.load_state_dict(torch.load(fk_path, map_location=device, weights_only=True))
        fk_ref.eval()
        for p in fk_ref.parameters():
            p.requires_grad = False

    X_train, X_test, y_train, y_test = split_data(data_df, in_cols, out_cols)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    # 1. 输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/learn_model") / f"{model_type}_{mode}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. 构造模型
    if model_type == 'mlp':
        model = FlexibleMLP(X_train.shape[1], y_train.shape[1],
                            hidden_layers,
                            dropout=0.0,  # 0.0% dropout
                            activation='ReLU',
                            block_type='res',
                            num_blocks=1).to(device)  # 换激活函数
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=min_lr)  # eta_min 最低学习率
        history = {'train': [], 'test': [], 'metric': 'MSE'}
        best_test = np.inf
        patience = 0
        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            y_pred = model(X_train)
            # y_pred = torch.tensor(y_pred, dtype=torch.float32, device=device)
            loss, info = compute_fk_loss(y_pred, y_train, w_pos=w_pos, w_ori=w_ori) \
                if mode == 'fk' else \
                compute_ik_loss(y_pred, y_train, pose_true=X_train, fk_ref=fk_ref, w_pos=w_pos, w_ori=w_ori)
            loss.backward()
            opt.step()
            scheduler.step()  # 退火
            history['train'].append(loss.item())

            # 每 epoch 记录测试
            with torch.no_grad():
                test_loss, test_info = compute_fk_loss(model(X_test), y_test, w_pos=w_pos,
                                                       w_ori=w_ori) if mode == 'fk' else compute_ik_loss(
                    model(X_test), y_test, pose_true=X_test, fk_ref=fk_ref, w_pos=w_pos, w_ori=w_ori)
            history['test'].append(test_loss.item())
            if (epoch + 1) % max(1, epochs // 100) == 0 or epoch == epochs - 1:
                print(
                    f"[{model_type.upper()} {mode.upper()}] Epoch {epoch + 1}/{epochs} | Train: {loss.item():.6f} | Test: {test_loss.item():.6f} | Test info: {test_info}")

            if test_loss < best_test:
                best_test = test_loss
                patience = 0
                torch.save(model.state_dict(), out_dir / 'best_model.pt')
            else:
                patience += 1
                if patience >= 50:
                    print(f"Early stop at epoch {epoch + 1}")
                    break

        torch.save(model.state_dict(), out_dir / 'model.pt')

    # 3. 画图
    curve_png = out_dir / f"training_curve_{history['metric'].replace(' ', '_')}.png"
    # 去掉 nan 再画
    clean = {}
    for k in ['train', 'test']:
        ser = pd.Series(history[k])  # 含 nan 的序列
        ser = ser.ffill()  # 前一有效帧填充
        clean[k] = ser.values  # 转回 ndarray
    clean['metric'] = history['metric']
    plot_training_curves(clean, str(curve_png))

    print(f"✅ 训练完成！模型与曲线已保存到 → {out_dir}")
    return model, out_dir, str(out_dir / 'best_model.pt')  # ① 返回 FK 模型路径


if __name__ == "__main__":
    # 训练正逆运动学模型
    fk_model, fk_dir, fk_path = train_dofbot_model(data_path='../dataset/60000/dofbot_fk_60000_norm.csv',
                                                   model_type='mlp', mode='fk',
                                                   fk_out_cols=['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
                                                   epochs=2000, lr=1e-3, hidden_layers=[128, 128, 64])
    ik_model, ik_dir, ik_path = train_dofbot_model(data_path='../dataset/60000/dofbot_fk_60000_norm.csv',
                                                   model_type='mlp', mode='ik',
                                                   ik_in_cols=['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
                                                   epochs=2000, lr=1e-3, hidden_layers=[128, 128, 64], fk_path=fk_path,
                                                   fk_hidden_layers=[128, 128, 64])
    # # 训练正逆运动学模型
    # fk_model, fk_dir, fk_path = train_dofbot_model(data_path='dataset/60000/dofbot_fk_60000_norm.csv',
    #                                                model_type='mlp', mode='fk',
    #                                                fk_out_cols=['x', 'y', 'z'],
    #                                                epochs=2000, lr=1e-3, hidden_layers=[128, 128, 64])
    # ik_model, ik_dir, ik_path = train_dofbot_model(data_path='dataset/60000/dofbot_fk_60000_norm.csv',
    #                                                model_type='mlp', mode='ik',
    #                                                ik_in_cols=['x', 'y', 'z'],
    #                                                epochs=2000, lr=1e-3, hidden_layers=[128, 128, 64], fk_path=fk_path)
