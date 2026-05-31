import rospy
from gazebo_msgs.msg import ModelState
import numpy as np

rospy.init_node('test_pub')
pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
msg = ModelState()
state = np.zeros([4,1])
msg.pose.position.x = state[1]
msg.pose.position.y = state[3]
try:
    pub.publish(msg)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
