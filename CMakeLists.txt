cmake_minimum_required(VERSION 3.8)
project(braid_tools)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/FlydraObject.msg
  msg/FlydraMainbrainPacket.msg
  msg/FlydraMainbrainSuperPacket.msg
  DEPENDENCIES geometry_msgs builtin_interfaces
)

install(PROGRAMS
  nodes/braid_ros_listener.py
  nodes/braid_emulator.py
  nodes/braid_realtime_plotter.py
  nodes/braid_save_data_to_hdf5.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  plot_config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  braid_tools_mapping_rules.yaml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
