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

    # ---------- 3. 定义新的状态机 ----------
    PRE_GRASP_STATE = 0
    LOWER_TO_GRASP_STATE = 1
    CLOSE_GRIPPER_STATE = 2
    LIFT_STATE = 3
    MOVE_TO_TARGET_STATE = 4
    LOWER_TO_SET_STATE = 5
    OPEN_GRIPPER_STATE = 6
    RETRACT_STATE = 7  # (可选) 抬起手臂
    current_state = PRE_GRASP_STATE

    # ---------- 4. 定义更精确的偏移量 ----------
    # 物块中心 z=0.015, 物块顶面 z=0.03

    # 抓取准备：z = 0.015(center) + 0.045 = 0.06 (距顶面 3cm)
    obj_offset_pre_grasp = [-0.015, -0.015, 0.045]

    # 实际抓取：z = 0.015(center) + 0.018 = 0.033 (距顶面 3mm)
    obj_offset_actual_grasp = [-0.015, -0.015, 0.018]

    # 抬起高度：z = 0.015(center) + 0.145 = 0.16
    obj_offset_lift = [-0.015, -0.015, 0.145]

    # 放置准备：z = 0.015(target) + 0.045 = 0.06,修改成了0.07
    obj_offset_pre_set = [-0.015, 0.015, 0.05]

    # 实际放置：z = 0.015(target) + 0.018 = 0.033
    obj_offset_actual_set = [-0.015, 0.015, 0.018]

    # 闭环控制参数
    ARRIVAL_THRESHOLD = 0.025  # 夹取物块和放置到目标位置设置的阈值

    # 用于开环延时（夹爪开合）
    delay_counter = 0
    GRIPPER_DELAY_STEPS = 50  # 仿真步数，用于等待夹爪闭合

    # 存储目标姿态，以便在状态间传递
    target_joint_poses = None
    target_pos = None

    time.sleep(1.0)

    while not Reward:

        # 持续获取传感器数据
        block_pos, block_orn, block_euler = env.get_block_pose()
        target_place_pos = env.get_target_pose()
        actual_pos, _, _ = env.get_dofbot_pose()

        # print("current_state", current_state)

        # 状态机
        if current_state == PRE_GRASP_STATE:
            # 状态0: 移动到物块上方
            target_pos = np.array(block_pos) + np.array(obj_offset_pre_grasp)
            target_joint_poses, _ = env.dofbot_setInverseKine(target_pos)
            env.dofbot_control(target_joint_poses, GRIPPER_DEFAULT_ANGLE)

            if np.linalg.norm(target_pos - actual_pos) < ARRIVAL_THRESHOLD:
                print("Feedback: (0) Reached PRE_GRASP. -> (1) LOWERING")
                current_state = LOWER_TO_GRASP_STATE

        elif current_state == LOWER_TO_GRASP_STATE:
            # 状态1: 下降到抓取高度
            target_pos = np.array(block_pos) + np.array(obj_offset_actual_grasp)
            target_joint_poses, _ = env.dofbot_setInverseKine(target_pos)
            env.dofbot_control(target_joint_poses, GRIPPER_DEFAULT_ANGLE)

            if np.linalg.norm(target_pos - actual_pos) < ARRIVAL_THRESHOLD:
                print("Feedback: (1) LOWERED. -> (2) CLOSING GRIPPER")
                current_state = CLOSE_GRIPPER_STATE
                delay_counter = 0  # 重置延时计数器

        elif current_state == CLOSE_GRIPPER_STATE:
            # 状态2: 闭合夹爪 (开环延时)
            env.dofbot_control(target_joint_poses, GRIPPER_CLOSE_ANGLE)  # 保持位置，闭合夹爪
            delay_counter += 1

            if delay_counter > GRIPPER_DELAY_STEPS:
                print("Feedback: (2) GRIPPER CLOSED. -> (3) LIFTING")
                current_state = LIFT_STATE

        elif current_state == LIFT_STATE:
            # 状态3: 垂直抬起
            # 注意：我们用 *当前* 物块位置来计算抬起目标，以防物块在抓取时轻微移动
            target_pos = np.array(block_pos) + np.array(obj_offset_lift)
            target_joint_poses, _ = env.dofbot_setInverseKine(target_pos)
            env.dofbot_control(target_joint_poses, GRIPPER_CLOSE_ANGLE)  # 保持抓紧

            print("Feedback: (3) LIFTED. -> (4) MOVING TO TARGET")
            current_state = MOVE_TO_TARGET_STATE

        elif current_state == MOVE_TO_TARGET_STATE:
            # 状态4: 移动到目标点上方
            target_pos = np.array(target_place_pos) + np.array(obj_offset_pre_set)
            target_joint_poses, _ = env.dofbot_setInverseKine(target_pos)
            env.dofbot_control(target_joint_poses, GRIPPER_CLOSE_ANGLE)  # 保持抓紧

            if np.linalg.norm(target_pos - actual_pos) < ARRIVAL_THRESHOLD:
                print("Feedback: (4) Reached TARGET. -> (5) LOWERING TO SET")
                current_state = LOWER_TO_SET_STATE

        elif current_state == LOWER_TO_SET_STATE:
            # 状态5: 下降到放置高度
            target_pos = np.array(target_place_pos) + np.array(obj_offset_actual_set)
            target_joint_poses, _ = env.dofbot_setInverseKine(target_pos)
            env.dofbot_control(target_joint_poses, GRIPPER_CLOSE_ANGLE)

            if np.linalg.norm(target_pos - actual_pos) < ARRIVAL_THRESHOLD:
                print("Feedback: (5) LOWERED TO SET. -> (6) OPENING GRIPPER")
                current_state = OPEN_GRIPPER_STATE
                delay_counter = 0  # 重置延时计数器

        elif current_state == OPEN_GRIPPER_STATE:
            # 状态6: 松开夹爪 (开环延时)
            env.dofbot_control(target_joint_poses, GRIPPER_DEFAULT_ANGLE)  # 保持位置，松开夹爪
            delay_counter += 1

            if delay_counter > GRIPPER_DELAY_STEPS:
                print("Feedback: (6) GRIPPER OPENED.")


        # 检查是否最终完成
        Reward = env.reward()

    # ---------- 结束录制 ----------
    p.stopStateLogging(log_id)
    print("Task Completed!")