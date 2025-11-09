from dofbot import DofbotEnv
import numpy as np
import copy
import time, os, datetime
import pybullet as p

# ---------- 1. 准备保存目录 ----------
save_dir = "results/record"
os.makedirs(save_dir, exist_ok=True)
mp4_path = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4")

if __name__ == '__main__':
    env = DofbotEnv()
    env.reset()
    Reward = False

    # 2. 开始录制
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                 mp4_path,
                                 physicsClientId=env.physicsClient)

    '''
    constants here
    '''
    GRIPPER_DEFAULT_ANGLE = 20. / 180. * 3.1415
    GRIPPER_CLOSE_ANGLE = -20. / 180. * 3.1415

    # define state machine
    PRE_GRASP_STATE = 0
    GRASP_STATE = 1
    MOVE_STATE = 2
    SET_STATE = 3
    current_state = PRE_GRASP_STATE

    # print("object1.size: ", env._object1.size)  # → [0.03, 0.03, 0.03]  （半尺寸）
    obj_offset_grasp = [-0.015, -0.015, 0.045]
    obj_offset_move = [0, 0, 0.145]
    obj_offset_set = [-0.015, 0.015, 0.045]

    block_pos, block_orn, block_euler = env.get_block_pose()


    start_time = None

    time.sleep(1.0)
    num = 0
    state_num = 1700

    while not Reward:
        '''
        #获取物块位姿、目标位置和机械臂位姿，计算机器臂关节和夹爪角度，使得机械臂夹取绿色物块，放置到紫色区域。
        '''

        '''
        code here
        '''
        # 获取目标（紫色）位置
        target_place_pos = env.get_target_pose()

        if current_state == PRE_GRASP_STATE:
            # 状态0: 移动到抓取位置(物块)上方
            print("State: PRE_GRASP")
            # 目标位置 = 物块位置 + 向上偏移
            target_pos = np.array(block_pos) + np.array(obj_offset_grasp)
            # 计算逆解
            joint_poses, _ = env.dofbot_setInverseKine(target_pos)
            # 控制机械臂（夹爪张开）
            env.dofbot_control(joint_poses, GRIPPER_DEFAULT_ANGLE)

            # 等待 10 步仿真
            num += 2
            if num > state_num:
                current_state = GRASP_STATE
                num = 0

        elif current_state == GRASP_STATE:
            # 状态1: 下降并抓取
            print("State: GRASP")
            # 目标位置 = 物块位置 + 抓取偏移
            target_pos = np.array(block_pos) + np.array(obj_offset_grasp)
            # 计算逆解
            joint_poses, _ = env.dofbot_setInverseKine(target_pos)
            env.dofbot_control(joint_poses, GRIPPER_DEFAULT_ANGLE)
            if num <= state_num:
                num += 1
                # 1. 向下移动（夹爪保持张开）
            else:
                current_state = MOVE_STATE
                # 2. 闭合夹爪

        elif current_state == MOVE_STATE:
            # 状态2: 抬起物块并移动到目标点上方
            print("State: MOVE")

            if num <= state_num:
                # 1. 垂直向上抬起
                lift_pos = np.array(block_pos) + np.array(obj_offset_move)
                joint_poses, _ = env.dofbot_setInverseKine(lift_pos)
                env.dofbot_control(joint_poses, GRIPPER_CLOSE_ANGLE)  # 保持夹爪闭合
            else:
                # 2. 移动到目标点上方
                move_pos = np.array(target_place_pos) + np.array(obj_offset_set)
                joint_poses, _ = env.dofbot_setInverseKine(move_pos)  # 保持原始抓取姿态
                env.dofbot_control(joint_poses, GRIPPER_CLOSE_ANGLE)  # 保持夹爪闭合

            num += 1
            # 等待 10 步抬起 + 10 步移动
            if num > state_num * 2:
                current_state = SET_STATE
                num = 0

        elif current_state == SET_STATE:
            # 状态3: 下降并释放
            print("State: SET")
            env.dofbot_control(joint_poses, GRIPPER_DEFAULT_ANGLE)

            num += 1
            # 此后，机械臂会保持在“移开”状态，
            # while 循环会等待 env.reward() 返回 True (物块落到目标位置)
            if num > state_num * 3 + 10:  # 防止计数器溢出
                num = state_num * 3 + 1

        # 兜底：如果状态未知，保持当前姿势
        else:
            joint_poses, gripper_angle = env.get_dofbot_jointPoses()
            env.dofbot_control(joint_poses, gripper_angle)

        Reward = env.reward()

    # ---------- 3. 结束录制 ----------
    p.stopStateLogging(log_id)