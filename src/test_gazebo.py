import rospy
from gazebo_msgs.msg import ModelState
import numpy as np

rospy.init_node('test_pub2')
pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10, latch=True)
rospy.sleep(1.0)
msg = ModelState()
msg.model_name = "cylinderRobot"
msg.pose.position.x = 2.0
msg.pose.position.y = 2.0
pub.publish(msg)
