"""
机器人学课程 Dofbot 机械臂基于改进DH参数法的正 / 逆运动学建模
"""

# --------------------- 1. 导入常用库 ---------------------
import roboticstoolbox as rtb   # 机器人专用工具箱
import numpy as np              # 矩阵运算
import matplotlib.pyplot as plt # 导入绘图库
from mpl_toolkits.mplot3d import Axes3D # 导入3D绘图工具

# --------------------- 2. 常量定义 ---------------------
pi = 3.1415926          # 自己指定 π，方便后续打印保留 7 位小数
# 连杆长度（单位：m，与实物一致）
l1 = 0.1045             # 连杆1长度（基座→关节2）
l2 = 0.08285            # 连杆2长度（关节2→关节3）
l3 = 0.08285            # 连杆3长度（关节3→关节4）
l4 = 0.12842            # 连杆4长度（关节4→末端）

# ==============================================
# 用改进 DH 法建立机器人模型Demo
# ==============================================
# RevoluteMDH(a, alpha, d, offset)
# 默认 theta 为关节变量，因此只写常数项即可
'''
DH_demo = rtb.DHRobot(
    [
        rtb.RevoluteMDH(d=l1),                              # 关节1：绕 z 旋转，d 向上偏移 l1
        rtb.RevoluteMDH(alpha=-pi/2, offset=-pi/3),         # 关节2：x 向下扭转 90°，初始偏置 -90°
        rtb.RevoluteMDH(a=l2, offset = pi / 6),                              # 关节3：平移 l2
        rtb.RevoluteMDH(a=l3, offset=pi * 2 / 3),                 # 关节4：平移 l3，初始偏置 +90°
        rtb.RevoluteMDH(alpha=pi/2, d=l4)                   # 关节5：x 向上扭转 90°，末端延伸 l4
    ],
    name="DH_demo"       # 给机器人起个名字，打印时更直观
)

# 打印标准 DH 参数表（alpha、a、d、theta、offset）
print("========== DH_demo机器人 DH 参数 ==========")
print(DH_demo)

# --------------------- 零位验证 ---------------------
fkine_input0 = [0, 0, 0, 0, 0]          # 全部关节置 0
fkine_result0 = DH_demo.fkine(fkine_input0)
print("\n零位正解齐次变换矩阵:")
print(fkine_result0)
DH_demo.plot(q=fkine_input0, block=True) # 3D 可视化（阻塞模式）
'''


# ==============================================
# 仿真任务0、 用改进 DH 法建立Dofbot机器人模型
# ==============================================
# RevoluteMDH(a, alpha, d, offset)
# 默认 theta 为关节变量，因此只写常数项即可
dofbot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(d=l1),
        rtb.RevoluteMDH(alpha=-pi/2, offset=-pi/2),
        rtb.RevoluteMDH(a=l2),
        rtb.RevoluteMDH(a=l3, offset=pi/2),
        rtb.RevoluteMDH(alpha=pi/2, d=l4)
    ],
    name="Dofbot"
)
'''
# 打印标准 DH 参数表（alpha、a、d、theta、offset）
print("========== Dofbot机器人 DH 参数 ==========")
print(dofbot)

# --------------------- 4. Part0 零位验证 ---------------------
fkine_input0 = [0, 0, 0, 0, 0]          # 全部关节置 0
fkine_result0 = dofbot.fkine(fkine_input0)
print("\n零位正解齐次变换矩阵:")
print(fkine_result0)
dofbot.plot(q=fkine_input0, block=True) # 3D 可视化（阻塞模式）

# ==============================================
# 仿真任务1、 正运动学 —— 给出DH模型在以下 4 组关节角下的正运动学解
# ==============================================
# poses = [
#     [0., pi/3, pi/4, pi/5, 0.],            # demo
#     [pi/2, pi/5, pi/5, pi/5, pi],          # 1
#     [pi/3, pi/4, -pi/3, -pi/4, pi/2],      # 2
#     [-pi/2, pi/3, -2*pi/3, pi/3, pi/3]     # 3
# ]

# -------- 1.1 demo  pose ----------
q_demo = [0., pi/3, pi/4, pi/5, 0.]
T_demo = dofbot.fkine(q_demo)
print("\n========== Part1-0 (demo) 正解 ==========")
print(T_demo)
dofbot.plot(q=q_demo, block=True)

# -------- 1.2 pose 1 ----------
q_1 = [pi/2, pi/5, pi/5, pi/5, pi]
T_1 = dofbot.fkine(q_1)
print("\n========== Part1-1 (Pose 1) 正解 ==========")
print(T_1)
print("Pose 1 姿态仿真 (请关闭窗口后继续)...")
dofbot.plot(q=q_1, block=True)




# -------- 1.3 pose 2 ----------
q_2 = [pi/3, pi/4, -pi/3, -pi/4, pi/2]
T_2 = dofbot.fkine(q_2)
print("\n========== Part1-2 (Pose 2) 正解 ==========")
print(T_2)
print("Pose 2 姿态仿真 (请关闭窗口后继续)...")
dofbot.plot(q=q_2, block=True)





# -------- 1.4 pose 3 ----------
q_3 = [-pi/2, pi/3, -2*pi/3, pi/3, pi/3]
T_3 = dofbot.fkine(q_3)
print("\n========== Part1-3 (Pose 3) 正解 ==========")
print(T_3)
print("Pose 3 姿态仿真 (请关闭窗口后继续)...")
dofbot.plot(q=q_3, block=True)
'''




# ==============================================
# 仿真任务2、 逆运动学 —— 给出DH模型在以下 4 组笛卡尔空间姿态下的逆运动学解
# ==============================================
# targets = [
#     # demo
#     np.array([
#         [-1., 0., 0., 0.1],
#         [ 0., 1., 0., 0. ],
#         [ 0., 0.,-1.,-0.1],
#         [ 0., 0., 0., 1. ]
#     ]),
#     # 1
#     np.array([
#         [1., 0., 0., 0.1],
#         [0., 1., 0., 0. ],
#         [0., 0., 1., 0.1],
#         [0., 0., 0., 1. ]
#     ]),
#     # 2
#     np.array([
#         [cos(pi/3), 0.,-sin(pi/3), 0.2],
#         [0.,        1., 0.,        0. ],
#         [sin(pi/3), 0., cos(pi/3), 0.2],
#         [0.,        0., 0.,        1. ]
#     ]),
#     # 3
#     np.array([
#         [-0.866, -0.25,  -0.433, -0.03704],
#         [ 0.5,   -0.433, -0.75,  -0.06415],
#         [ 0.,    -0.866,  0.5,    0.3073 ],
#         [ 0.,     0.,     0.,     1.     ]
#     ])
# ]

'''
# -------- 2.1 demo 目标 ----------
T_des_demo = np.array([
    [-1., 0., 0., 0.1],
    [ 0., 1., 0., 0. ],
    [ 0., 0.,-1.,-0.1],
    [ 0., 0., 0., 1. ]
])
q_ik_demo = dofbot.ik_LM(T_des_demo)[0]   # 取返回元组第 0 个元素
print("\n========== Part2-0 (demo) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_demo))
dofbot.plot(q=q_ik_demo, block=True)

# -------- 2.2 目标 1 ----------
T_des_1 = np.array([
    [1., 0., 0., 0.1],
    [0., 1., 0., 0. ],
    [0., 0., 1., 0.1],
    [0., 0., 0., 1. ]
])
q_ik_1 = dofbot.ik_LM(T_des_1)[0]
print("\n========== Part2-1 (目标 1) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_1))
dofbot.plot(q=q_ik_1, block=True)

# -------- 2.3 目标 2 ----------
T_des_2 = np.array([
    [np.cos(pi/3), 0.,-np.sin(pi/3), 0.2],
    [0.,           1., 0.,           0. ],
    [np.sin(pi/3), 0., np.cos(pi/3), 0.2],
    [0.,           0., 0.,           1. ]
])
q_ik_2 = dofbot.ik_LM(T_des_2)[0]
print("\n========== Part2-2 (目标 2) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_2))
dofbot.plot(q=q_ik_2, block=True)

# -------- 2.4 目标 3 ----------
T_des_3 = np.array([
    [-0.866, -0.25,  -0.433, -0.03704],
    [ 0.5,    -0.433, -0.75,  -0.06415],
    [ 0.,     -0.866,  0.5,    0.3073 ],
    [ 0.,      0.,     0.,     1.     ]
])
q_ik_3 = dofbot.ik_LM(T_des_3)[0]
print("\n========== Part2-3 (目标 3) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_3))
dofbot.plot(q=q_ik_3, block=True)
'''
# ==============================================
# 仿真任务3、 工作空间可视化（≥500 点）
#     关节限位（°）→ 弧度
#     J1: [-180, 180]  J2~J5: [0, 180]
# ==============================================

# 关节角度限位 (来自图像, 单位：弧度)
qlim_j1 = [-pi, pi]       # J1: [-180°, 180°]
qlim_j2 = [0, pi]         # J2: [0°, 180°]
qlim_j3 = [0, pi]         # J3: [0°, 180°]
qlim_j4 = [0, pi]         # J4: [0°, 180°]
qlim_j5 = [0, pi]         # J5: [0°, 180°]

# 使用更新的关节限位 (qlim) 来定义 Dofbot
dofbot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(d=l1, qlim=qlim_j1),
        rtb.RevoluteMDH(alpha=-pi/2, offset=-pi/2, qlim=qlim_j2),
        rtb.RevoluteMDH(a=l2, qlim=qlim_j3),
        rtb.RevoluteMDH(a=l3, offset=pi/2, qlim=qlim_j4),
        rtb.RevoluteMDH(alpha=pi/2, d=l4, qlim=qlim_j5)
    ],
    name="Dofbot"
)

# 采样点数 (示例图为 8000, 任务要求至少 500)
N = 8000

# 1. 定义关节角度范围 (与图像一致, 单位：度)
#    我们在这里用“度”更直观，在循环中转换为弧度
joint_limits_deg = [
    [-180, 180],  # J1
    [0, 180],  # J2
    [0, 180],  # J3
    [0, 180],  # J4
    [0, 180]  # J5
]

# 采样点容器
workspace_points = []

print(f"检测到旧版库，正在为 {N} 个采样点循环计算正运动学...")

# 2. 循环生成随机角度并计算正运动学
for i in range(N):
    # 为每个关节生成随机角度 (并转换为弧度)
    q = [
        np.random.uniform(limit[0], limit[1]) * pi / 180
        for limit in joint_limits_deg
    ]

    # 计算末端位置
    # .fkine() 在旧版中一次只处理一组q
    pose = dofbot.fkine(q)
    workspace_points.append(pose.t)  # 获取位姿中的位置部分

# 3. 将点列表转换为 NumPy 数组 (N, 3)
points = np.array(workspace_points)

print(f"计算完成。正在生成 {N} 个点的工作空间散点图...")
# -----------------------------------------------------------------
# 【代码修正结束】


# 4. 3D 散点图可视化 (这部分代码无需改变)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.3)

# 设置坐标轴标签和标题
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'Dofbot Workspace (N={N})')

# --- 可选：让 3D 坐标轴的比例看起来更均匀 ---
max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                      points[:, 1].max() - points[:, 1].min(),
                      points[:, 2].max() - points[:, 2].min()]).max() / 2.0

mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
# ----------------------------------------

print("绘图完成。")
plt.show()