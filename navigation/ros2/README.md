# ROS 2 navigation dependencies

ActuateX always publishes a standard `sensor_msgs/PointCloud2` whose packed
fields match the official `livox_ros_driver2` `PointXYZRTLT` layout. No vendor
message package is required for that default path.

To also publish the official `livox_ros_driver2/CustomMsg`, import and build the
pinned upstream package in the ROS workspace used by the simulator:

```bash
vcs import src < navigation/ros2/dependencies.repos
colcon build --packages-select livox_ros_driver2 actuatex_navigation
source install/setup.bash
```

The pin is deliberate: message ABI and field order must not silently drift.
The upstream code remains in the ROS workspace and is not vendored here. If
the package is absent, the simulator logs a warning and continues publishing
the exact 26-byte PointCloud2 representation plus the Nav2 `/scan` projection.

Isaac Sim 6.0.1 bundles Python 3.12-compatible ROS 2 libraries; do not inject
Ubuntu 22.04 Humble's Python 3.10 `rclpy` into the simulator process. The
`run_tinymal_nav2.py` launcher strips incompatible system paths before Kit
starts. External Nav2 nodes can still run in their normal ROS environment and
communicate over DDS.
