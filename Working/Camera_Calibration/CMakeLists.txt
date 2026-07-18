cmake_minimum_required(VERSION 3.0.2)
project(camera_calibration)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2_ros
)

catkin_package()

# Install Python scripts to the catkin bin directory
catkin_install_python(PROGRAMS
  calibrate_d405_hand_eye.py
  calibrate_d435_bird_eye.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# Install calibration config
install(FILES
  calibration_config.yaml
  calibration_math.py
  __init__.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
