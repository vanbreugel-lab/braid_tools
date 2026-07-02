# Braid Tools

* Helpful nodes for interfacing Braid with ROS.
* Basic analysis, slicing, and plotting functions.

# ROS nodes

### braid_ros_listener.py

Node that listens to an active Braid instance and streams the data to a ROS topic: `flydra_mainbrain/super_packets`. To run:
* `rosrun braid_tools braid_ros_listener.py --braid-model-server-url=http://YOUR.IP.ADDRESS:8397/`

### braid_emulator.py

Simulates a moving trajectory and outputs in the same format as the braid_ros_listener.  Helpful for testing hardware without needing real trajectory data. 

### braid_realtime_plotter.py

Allows you to visualize where in space tracked objects are.  Helpful for trouble shooting orientations of calibrations and testing tracking. Only supports a single object (first object). Uses a yaml file to set plotting dimensions, modify to fit your arena. To run:
* `rosrun braid_tools braid_realtime_plotter.py --config='~/catkin_ws/src/braid_tools/plot_config/small_tunnel.yaml'` 

### braid_save_data_to_hdf5.py

ROS node for saving `flydra_mainbrain/super_packets` to an hdf5 file with buffering etc. 

### topic_relay_server.py

Relays a configurable list of ROS 1 topics to ROS 2 machines over HTTP (server-sent events with JSON, the same pattern Braid uses) — a lightweight alternative to ros1_bridge. The matching client (`topic_relay_client.py`) lives on the `ros2` branch of this repo. To run:
* `rosrun braid_tools topic_relay_server.py --config $(rospack find braid_tools)/relay_config/relay_topics.yaml`

Edit `relay_config/relay_topics.yaml` to choose which topics to relay; message types are auto-detected from the ROS master. Serves on port 8398 by default. Intended for telemetry-sized messages (tracking packets, triggers, floats) — do not relay images or point clouds.

# Analysis

From inside this directory, run `python ./setup.py install` to install the analysis tools. 

See `examples` for demos. 
