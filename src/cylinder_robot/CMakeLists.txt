cmake_minimum_required(VERSION 3.0.2)
project(cylinder_robot)

find_package(catkin REQUIRED COMPONENTS
  gazebo_msgs
  geometry_msgs
  rospy
  std_msgs
)

catkin_package()

# 这里的路径对应你 script 文件夹里的文件名
catkin_install_python(PROGRAMS
  script/controller.py
  script/driver.py
  script/perception.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
