cmake_minimum_required(VERSION 2.8.3)
project(braid_tools)

find_package(catkin REQUIRED COMPONENTS
    ros_flydra
    roscpp
    rospy
    std_msgs
    message_generation
    std_msgs
    sensor_msgs
    geometry_msgs
    dynamic_reconfigure
)

find_package(
    catkin
    REQUIRED COMPONENTS
    message_generation
    std_msgs
    sensor_msgs
    geometry_msgs
    dynamic_reconfigure
)

#generate_messages(
#  DEPENDENCIES std_msgs geometry_msgs
#)

catkin_package( ## if you dont add this, executables are not found
  INCLUDE_DIRS nodes
#  LIBRARIES pcl_tutorials
  CATKIN_DEPENDS geometry_msgs nav_msgs pcl_msgs roscpp rospy std_msgs ros_flydra
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY
  launch/
  nodes/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.launch"
  PATTERN ".svn" EXCLUDE
)

install(PROGRAMS
  nodes/braid_ros_listener.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
