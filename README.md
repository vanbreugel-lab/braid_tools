# Braid Tools

* Helpful nodes for interfacing Braid with ROS 2.
* Basic analysis, slicing, and plotting functions.

This branch (`ros2`) targets **ROS 2 Jazzy** on Ubuntu 24.04. The `main` branch contains the original ROS 1 (catkin) version.

# Building

From your ROS 2 workspace root (e.g. `~/ros2_ws`):

```bash
colcon build --packages-select braid_tools
source install/setup.bash
```

Python dependencies for the nodes come from apt (`python3-requests`, `python3-h5py`, `python3-matplotlib`, `python3-yaml`, `python3-numpy`; `rosdep install` will pick them up from package.xml). The emulator optionally uses `pynumdiff` for velocity estimation, which is pip-only — on Ubuntu 24.04 install it in a venv or with `pip install --break-system-packages pynumdiff`; without it the emulator falls back to `np.gradient`.

# ROS nodes

### braid_ros_listener.py

Node that listens to an active Braid instance and streams the data to a ROS topic: `flydra_mainbrain/super_packets`. To run:
* `ros2 run braid_tools braid_ros_listener.py --braid-model-server-url=http://YOUR.IP.ADDRESS:8397/`

Or via launch file (defaults to the lab Braid server):
* `ros2 launch braid_tools braid_listener.launch.py braid_url:=http://YOUR.IP.ADDRESS:8397/`

### braid_emulator.py

Simulates a moving trajectory and outputs in the same format as the braid_ros_listener. Helpful for testing hardware without needing real trajectory data.
* `ros2 run braid_tools braid_emulator.py --fps 100`

### braid_realtime_plotter.py

Allows you to visualize where in space tracked objects are. Helpful for trouble shooting orientations of calibrations and testing tracking. Only supports a single object (first object). Uses a yaml file to set plotting dimensions, modify to fit your arena. To run:
* `ros2 run braid_tools braid_realtime_plotter.py --config=/path/to/plot_config/small_tunnel.yaml`

If `--config` is omitted, the installed `plot_config/small_tunnel.yaml` is used. There is also a combined launch file:
* `ros2 launch braid_tools braid_listener_and_live_plotter.launch.py plot_config:=/path/to/big_tunnel.yaml`

### braid_save_data_to_hdf5.py

ROS node for saving `flydra_mainbrain/super_packets` to an hdf5 file with buffering etc.
* `ros2 run braid_tools braid_save_data_to_hdf5.py --ros-args -p data_directory:=/path/to/save/dir`

The save directory can be given either with the `data_directory` ROS parameter (shown above) or the `--home-directory` CLI argument; defaults to `~/Desktop/temp`.

# ROS 1 interoperability

## Option 1: topic relay (recommended, simple)

A lightweight relay streams ROS 1 topics to ROS 2 over HTTP the same way Braid streams tracking data — no ros1_bridge, no Docker.

* **On the ROS 1 machine** (`main` branch of this repo): run the relay server with a yaml listing the topics to relay (see `relay_config/relay_topics.yaml`):
  ```bash
  rosrun braid_tools topic_relay_server.py --config $(rospack find braid_tools)/relay_config/relay_topics.yaml
  ```
* **On the ROS 2 machine** (this branch): run the client pointed at the ROS 1 machine:
  ```bash
  ros2 run braid_tools topic_relay_client.py --url http://ROS1.MACHINE.IP:8398/
  ```

The client needs no configuration — the server announces which topics and message types it relays, and the client republishes them under the same topic names. Message types are auto-detected on the ROS 1 side; only types whose ROS 2 name differs from `pkg/msg/Name` need an entry in the yaml's `type_map` (the braid_tools messages are pre-mapped). The client reconnects automatically if the server restarts. Intended for telemetry-sized messages (tracking packets, triggers, floats) — do not relay images or point clouds.

## Option 2: ros1_bridge

For full bidirectional bridging, `ros1_bridge` works but must be source-built. The following matter on this package's side:

* The package name (`braid_tools`) and message field names are identical on both branches, but ROS 2 requires CamelCase message names (`flydra_object` → `FlydraObject`, etc.). The mapping is declared in `braid_tools_mapping_rules.yaml`, which is installed to the package share directory and exported for the bridge.
* Custom messages always require a **source-built** ros1_bridge whose ROS 1 underlay includes the `main` branch build of this package and whose ROS 2 underlay includes the `ros2` branch build.
* There is no official ros1_bridge release for Jazzy/24.04. The usual setup is a Noetic + Humble bridge host (or container); Jazzy nodes interoperate with the Humble side over DDS since both build identical message definitions from this repo.
* The nodes use default QoS (reliable/volatile) and the plain topic name `flydra_mainbrain/super_packets`, both of which the bridge handles automatically.
* ROS 1 `time` fields map automatically to `builtin_interfaces/Time`.

# Analysis

The `braid_analysis` Python library is ROS-independent and unchanged from the `main` branch. From inside this directory, run `python ./setup.py install` to install the analysis tools.

See `example` for demos.
