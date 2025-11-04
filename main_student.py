from dofbot import DofbotEnv
import numpy as np
import copy
import time, os, datetime
import pybullet as p

# # ---------- 1. 准备保存目录 ----------
# save_dir = "results/record"
# os.makedirs(save_dir, exist_ok=True)
# mp4_path = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4")

if __name__ == '__main__':
    env = DofbotEnv()
    env.reset()
    Reward = False

    # # 2. 开始录制
    # log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
    #                              mp4_path,
    #                              physicsClientId=env.physicsClient)

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
    state_num = 10

    while not Reward:
        '''
        #获取物块位姿、目标位置和机械臂位姿，计算机器臂关节和夹爪角度，使得机械臂夹取绿色物块，放置到紫色区域。
        '''

        '''
        code here
        '''








        Reward = env.reward()

    # # ---------- 3. 结束录制 ----------
    # p.stopStateLogging(log_id)