#!/usr/bin/env python
"""Run a TinyMal locomotion policy as the mobile base controlled by Nav2."""

from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import platform
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
NAVIGATION_PYTHON_ROOT = PROJECT_ROOT / "navigation" / "ros2" / "actuatex_navigation"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(NAVIGATION_PYTHON_ROOT))


def _is_system_ros_path(path: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to("/opt/ros")
    except (OSError, ValueError):
        return False


def _select_isaac_ros_python() -> None:
    """Force Isaac Sim's Python-3.12-compatible bundled ROS 2 libraries."""

    sys.path[:] = [path for path in sys.path if not _is_system_ros_path(path)]
    for variable in (
        "PYTHONPATH",
        "OLD_PYTHONPATH",
        "LD_LIBRARY_PATH",
        "AMENT_PREFIX_PATH",
    ):
        paths = os.environ.get(variable, "").split(os.pathsep)
        compatible = [path for path in paths if path and not _is_system_ros_path(path)]
        if compatible:
            os.environ[variable] = os.pathsep.join(compatible)
        else:
            os.environ.pop(variable, None)
    isaac_root_text = os.environ.get("ISAAC_PATH")
    if not isaac_root_text:
        raise RuntimeError(
            "ISAAC_PATH is unset; launch this script through Isaac Lab's isaaclab.sh"
        )
    isaac_root = Path(isaac_root_text).resolve()
    ros_core_root = isaac_root / "exts" / "isaacsim.ros2.core"
    ubuntu_major = platform.freedesktop_os_release().get("VERSION_ID", "").split(".")[0]
    ros_distro = {"22": "humble", "24": "jazzy"}.get(ubuntu_major)
    if ros_distro is None:
        raise RuntimeError(
            f"Isaac Sim 6 bundled ROS 2 is unsupported on Ubuntu {ubuntu_major!r}"
        )
    ros_library_root = ros_core_root / ros_distro / "lib"
    ros_python_root = ros_core_root / ros_distro / "rclpy"
    if not ros_library_root.is_dir() or not ros_python_root.is_dir():
        raise FileNotFoundError(
            f"incomplete Isaac Sim bundled ROS 2 tree: {ros_core_root / ros_distro}"
        )

    # setup_ros_env.sh must take effect before the ELF loader starts Python;
    # changing LD_LIBRARY_PATH after startup is too late for ROS backend
    # dependencies.  Re-exec the same Sim Python process exactly once with the
    # official bundled library and Python roots selected.
    for variable, path in (
        ("LD_LIBRARY_PATH", ros_library_root),
        ("PYTHONPATH", ros_python_root),
    ):
        existing = [
            entry for entry in os.environ.get(variable, "").split(os.pathsep) if entry
        ]
        path_text = str(path)
        os.environ[variable] = os.pathsep.join(
            [path_text, *[entry for entry in existing if entry != path_text]]
        )
    os.environ["ROS_DISTRO"] = ros_distro
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    if os.environ.get("ACTUATEX_ROS2_ENV_READY") != "1":
        os.environ["ACTUATEX_ROS2_ENV_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ.copy())


_select_isaac_ros_python()

from actuatex_navigation.camera_profile import (  # noqa: E402
    camera_frame_quaternions,
    load_camera_calibration,
)
from actuatex_navigation.mid360_pattern import (  # noqa: E402
    load_mid360_pattern,
)
from actuatex_navigation.mid360_ros import Mid360RosPublisher  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(
    description="Bridge ROS 2 Nav2 velocity commands into a TinyMal policy."
)
parser.add_argument(
    "--actor", required=True, help="TorchScript actor or RSL-RL checkpoint"
)
parser.add_argument("--command_topic", default="/actuatex/policy_cmd")
parser.add_argument("--odom_topic", default="/odom")
parser.add_argument("--scan_topic", default="/scan")
parser.add_argument(
    "--pointcloud_topic",
    default="/livox/lidar",
    help="official livox_ros_driver2 PointXYZRTLT-compatible PointCloud2 topic",
)
parser.add_argument(
    "--livox_custom_topic",
    default="/livox/lidar_custom",
    help=(
        "optional official livox_ros_driver2/CustomMsg topic; empty disables it, "
        "and a missing livox_ros_driver2 package falls back to PointCloud2"
    ),
)
parser.add_argument("--odom_frame", default="odom")
parser.add_argument("--base_frame", default="base_link")
parser.add_argument("--scan_frame", default="base_scan")
parser.add_argument(
    "--lidar_type",
    choices=("mid360", "planar"),
    default="mid360",
    help="use the point-timed Livox model or the legacy rotary 2D control",
)
parser.add_argument("--lidar_height", type=float, default=0.12)
parser.add_argument(
    "--mid360_pattern",
    type=Path,
    help="compatible 800,000-row Livox CSV or CSV.xz; default uses pinned data",
)
parser.add_argument(
    "--mid360_stride",
    type=int,
    default=1,
    help="retain every Nth firing; one is the exact 200 kpoint/s path",
)
parser.add_argument(
    "--mid360_mount_xyz",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.0, 0.0, 0.12),
    help="Mid-360 optical-origin translation in the base frame, metres",
)
parser.add_argument(
    "--mid360_mount_rpy_deg",
    type=float,
    nargs=3,
    metavar=("ROLL", "PITCH", "YAW"),
    default=(0.0, 0.0, 0.0),
    help="fixed-axis Mid-360 mount rotation in REP-103 base coordinates",
)
parser.add_argument(
    "--mid360_motion_compensation",
    choices=("raw", "compensated"),
    default="raw",
    help="raw preserves physical acquisition skew; compensated is an A/B mode",
)
parser.add_argument(
    "--mid360_visual",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="attach the open 65 x 65 x 60 mm visual teaching shell",
)
parser.add_argument("--mid360_angular_error_std_deg", type=float, default=0.05)
parser.add_argument("--mid360_range_accuracy_m", type=float, default=0.02)
parser.add_argument("--scan_bins", type=int, default=1440)
parser.add_argument("--scan_min_z", type=float, default=-0.08)
parser.add_argument("--scan_max_z", type=float, default=0.08)
parser.add_argument(
    "--camera_calibration",
    type=Path,
    help=(
        "ROS camera_info YAML; supplying it enables the Isaac Sim 6 RTX camera. "
        "No uncalibrated camera is created by default."
    ),
)
parser.add_argument("--camera_topic", default="/camera/color/image_raw")
parser.add_argument("--camera_info_topic", default="/camera/color/camera_info")
parser.add_argument(
    "--camera_depth_topic",
    default="",
    help="optional ideal depth topic; empty disables depth publication",
)
parser.add_argument("--camera_frame", default="front_camera_optical_frame")
parser.add_argument("--camera_rate", type=float, default=30.0)
parser.add_argument(
    "--camera_mount_xyz",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.16, 0.0, 0.08),
    help="camera optical-center translation in the robot base frame, metres",
)
parser.add_argument(
    "--camera_mount_rpy_deg",
    type=float,
    nargs=3,
    metavar=("ROLL", "PITCH", "YAW"),
    default=(0.0, 0.0, 0.0),
    help="fixed-axis camera mount rotation in REP-103 base coordinates",
)
parser.add_argument("--camera_exposure_time", type=float, default=0.005)
parser.add_argument("--camera_f_stop", type=float, default=2.8)
parser.add_argument("--camera_responsivity", type=float, default=1.0)
parser.add_argument("--camera_focus_distance", type=float, default=2.0)
parser.add_argument("--camera_near", type=float, default=0.05)
parser.add_argument("--camera_far", type=float, default=50.0)
parser.add_argument("--command_timeout", type=float, default=0.35)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="stop after this many 50 Hz policy steps; zero runs until interrupted",
)
parser.add_argument(
    "--continue_after_fall",
    action="store_true",
    help="allow Isaac Lab to reset and continue after a termination",
)
parser.add_argument("--ros_domain_id", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.command_timeout <= 0.0:
    parser.error("--command_timeout must be positive")
if args_cli.max_steps < 0:
    parser.error("--max_steps must be non-negative")
if args_cli.lidar_height <= 0.0:
    parser.error("--lidar_height must be positive")
if args_cli.mid360_stride < 1:
    parser.error("--mid360_stride must be at least one")
if args_cli.mid360_angular_error_std_deg < 0.0:
    parser.error("--mid360_angular_error_std_deg cannot be negative")
if args_cli.mid360_range_accuracy_m < 0.0:
    parser.error("--mid360_range_accuracy_m cannot be negative")
if args_cli.scan_bins < 8:
    parser.error("--scan_bins must be at least eight")
if args_cli.scan_min_z >= args_cli.scan_max_z:
    parser.error("--scan_min_z must be smaller than --scan_max_z")
if args_cli.lidar_type == "mid360" and not args_cli.pointcloud_topic:
    parser.error("--pointcloud_topic cannot be empty for the Mid-360")
if args_cli.camera_rate <= 0.0:
    parser.error("--camera_rate must be positive")
if args_cli.camera_exposure_time <= 0.0:
    parser.error("--camera_exposure_time must be positive")
if args_cli.camera_exposure_time > 1.0 / args_cli.camera_rate:
    parser.error("--camera_exposure_time cannot exceed the camera frame period")
if args_cli.camera_f_stop <= 0.0 or args_cli.camera_responsivity <= 0.0:
    parser.error("camera f-stop and responsivity must be positive")
if args_cli.camera_focus_distance <= 0.0:
    parser.error("--camera_focus_distance must be positive")
if args_cli.camera_near <= 0.0 or args_cli.camera_far <= args_cli.camera_near:
    parser.error("camera clipping range must satisfy 0 < near < far")
if args_cli.camera_depth_topic and args_cli.camera_calibration is None:
    parser.error("--camera_depth_topic requires --camera_calibration")
if args_cli.ros_domain_id is not None:
    os.environ["ROS_DOMAIN_ID"] = str(args_cli.ros_domain_id)

camera_calibration = None
camera_usd_wxyz = None
camera_ros_xyzw = None
mid360_pattern = None
if args_cli.lidar_type == "mid360":
    try:
        mid360_pattern = load_mid360_pattern(args_cli.mid360_pattern)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(f"invalid Mid-360 firing pattern: {exc}")
if args_cli.camera_calibration is not None:
    try:
        camera_calibration = load_camera_calibration(args_cli.camera_calibration)
        camera_usd_wxyz, camera_ros_xyzw = camera_frame_quaternions(
            args_cli.camera_mount_rpy_deg
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(f"invalid camera calibration: {exc}")
args_cli.enable_cameras = True
kit_args = (getattr(args_cli, "kit_args", "") or "").strip()
required_kit_args = (
    "--enable omni.usd.schema.omni_lens_distortion",
    "--enable isaacsim.sensors.experimental.rtx",
    "--enable isaacsim.sensors.rtx.nodes",
    "--enable isaacsim.ros2.bridge",
    "--enable isaacsim.ros2.nodes",
    "--/renderer/raytracingMotion/enabled=true",
    "--/rtx/hydra/supportMultiTickRate=true",
    "--/rtx/rendering/perSensorTickTlas=true",
)
for required_arg in required_kit_args:
    if required_arg not in kit_args:
        kit_args = f"{kit_args} {required_arg}".strip()
args_cli.kit_args = kit_args

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni  # noqa: E402
import omni.syntheticdata  # noqa: E402
import omni.syntheticdata._syntheticdata as sd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tinymal_lab  # noqa: E402,F401
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402

# Keep this idempotent fallback for custom Kit experiences.  The normal path
# enables these through ``kit_args`` above so the lens schema is registered
# before USD initializes.
app_utils.enable_extension("isaacsim.sensors.experimental.rtx")
app_utils.enable_extension("isaacsim.sensors.rtx.nodes")
app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("isaacsim.ros2.nodes")
simulation_app.update()

from isaacsim.sensors.experimental.rtx import (  # noqa: E402
    CameraSensor,
    Lidar,
    LidarSensor,
    RtxCamera,
)
from pxr import Gf, Sdf  # noqa: E402
from tinymal_lab.mid360_rtx import (  # noqa: E402
    Mid360Runtime,
    create_mid360,
    update_mid360_pose,
)
from tinymal_lab.tinymal_navigation_env_cfg import (  # noqa: E402
    TinymalNavigationEnvCfg,
)

# These modules are supplied by the enabled Isaac Sim bridge, not by the
# system ROS Python installation.
import rclpy  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import TransformStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster  # noqa: E402


OBSERVATION_COMMAND_SCALE = (2.0, 2.0, 0.25)


def _pump_kit_for_rtx() -> None:
    """Render RTX products without allowing Kit to double-step physics."""

    settings = carb.settings.get_settings()
    setting_path = "/app/player/playSimulations"
    previous = settings.get(setting_path)
    settings.set_bool(setting_path, False)
    try:
        simulation_app.update()
    finally:
        settings.set_bool(setting_path, True if previous is None else bool(previous))


def _rpy_deg_to_wxyz(
    rpy_deg: tuple[float, float, float] | list[float],
) -> tuple[float, float, float, float]:
    """Convert fixed-axis REP-103 roll/pitch/yaw into a WXYZ quaternion."""

    roll, pitch, yaw = (math.radians(float(value)) for value in rpy_deg)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _build_actor() -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = 48
    for width in (512, 256, 128):
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, 12))
    return nn.Sequential(*layers)


def _actor_state(payload: dict) -> dict[str, torch.Tensor]:
    if "actor_state_dict" in payload:
        return {
            key[len("mlp.") :]: value
            for key, value in payload["actor_state_dict"].items()
            if key.startswith("mlp.")
        }
    if "model_state_dict" in payload:
        return {
            key[len("actor.") :]: value
            for key, value in payload["model_state_dict"].items()
            if key.startswith("actor.")
        }
    raise KeyError("checkpoint has neither actor_state_dict nor model_state_dict")


def load_actor(path: str, device: str) -> nn.Module:
    """Load the exported actor first, with a checkpoint fallback for teaching."""

    actor_path = os.path.abspath(path)
    if not os.path.isfile(actor_path):
        raise FileNotFoundError(actor_path)
    try:
        actor = torch.jit.load(actor_path, map_location=device)
        actor.eval()
    except RuntimeError:
        payload = torch.load(actor_path, map_location="cpu", weights_only=False)
        actor = _build_actor().to(device)
        actor.load_state_dict(_actor_state(payload), strict=True)
        actor.eval()

    with torch.no_grad():
        output = actor(torch.zeros(1, 48, device=device))
    if tuple(output.shape) != (1, 12):
        raise ValueError(
            f"actor must map [1, 48] to [1, 12], got {tuple(output.shape)}"
        )
    return actor


def _time_message(sim_time_ns: int) -> Time:
    stamp = Time()
    stamp.sec = sim_time_ns // 1_000_000_000
    stamp.nanosec = sim_time_ns % 1_000_000_000
    return stamp


class Nav2PolicyBridge(Node):
    """ROS I/O and a second, simulation-time command watchdog."""

    def __init__(self) -> None:
        super().__init__("tinymal_policy_bridge")
        self._command = (0.0, 0.0, 0.0)
        self._sim_time_ns = 0
        self._last_command_ns: int | None = None
        self._timeout_ns = round(args_cli.command_timeout * 1.0e9)
        self._odom_publisher = self.create_publisher(Odometry, args_cli.odom_topic, 20)
        self._clock_publisher = self.create_publisher(Clock, "/clock", 10)
        self._command_subscription = self.create_subscription(
            Twist, args_cli.command_topic, self._command_callback, 10
        )
        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._publish_static_transform()
        self.get_logger().info(
            f"policy command: {args_cli.command_topic}; odometry: {args_cli.odom_topic}"
        )

    def _command_callback(self, message: Twist) -> None:
        self._command = (
            float(message.linear.x),
            float(message.linear.y),
            float(message.angular.z),
        )
        self._last_command_ns = self._sim_time_ns

    def set_sim_time(self, sim_time_ns: int) -> None:
        self._sim_time_ns = sim_time_ns

    def command(self) -> tuple[float, float, float]:
        if (
            self._last_command_ns is None
            or self._sim_time_ns - self._last_command_ns > self._timeout_ns
        ):
            return (0.0, 0.0, 0.0)
        return self._command

    def publish_clock(self) -> None:
        message = Clock()
        message.clock = _time_message(self._sim_time_ns)
        self._clock_publisher.publish(message)

    def _publish_static_transform(self) -> None:
        lidar_transform = TransformStamped()
        lidar_transform.header.stamp = _time_message(self._sim_time_ns)
        lidar_transform.header.frame_id = args_cli.base_frame
        lidar_transform.child_frame_id = args_cli.scan_frame
        if args_cli.lidar_type == "mid360":
            lidar_transform.transform.translation.x = args_cli.mid360_mount_xyz[0]
            lidar_transform.transform.translation.y = args_cli.mid360_mount_xyz[1]
            lidar_transform.transform.translation.z = args_cli.mid360_mount_xyz[2]
            w, x, y, z = _rpy_deg_to_wxyz(args_cli.mid360_mount_rpy_deg)
            lidar_transform.transform.rotation.x = x
            lidar_transform.transform.rotation.y = y
            lidar_transform.transform.rotation.z = z
            lidar_transform.transform.rotation.w = w
        else:
            lidar_transform.transform.translation.z = args_cli.lidar_height
            lidar_transform.transform.rotation.w = 1.0
        transforms = [lidar_transform]

        if camera_calibration is not None:
            camera_transform = TransformStamped()
            camera_transform.header.stamp = _time_message(self._sim_time_ns)
            camera_transform.header.frame_id = args_cli.base_frame
            camera_transform.child_frame_id = args_cli.camera_frame
            camera_transform.transform.translation.x = args_cli.camera_mount_xyz[0]
            camera_transform.transform.translation.y = args_cli.camera_mount_xyz[1]
            camera_transform.transform.translation.z = args_cli.camera_mount_xyz[2]
            camera_transform.transform.rotation.x = camera_ros_xyzw[0]
            camera_transform.transform.rotation.y = camera_ros_xyzw[1]
            camera_transform.transform.rotation.z = camera_ros_xyzw[2]
            camera_transform.transform.rotation.w = camera_ros_xyzw[3]
            transforms.append(camera_transform)
        self._static_tf.sendTransform(transforms)

    def publish_robot_state(self, robot_data) -> None:
        position = robot_data.root_pos_w.torch[0].detach().cpu().tolist()
        quaternion = robot_data.root_quat_w.torch[0].detach().cpu().tolist()
        linear_velocity = robot_data.root_lin_vel_b.torch[0].detach().cpu().tolist()
        angular_velocity = robot_data.root_ang_vel_b.torch[0].detach().cpu().tolist()
        stamp = _time_message(self._sim_time_ns)

        odometry = Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = args_cli.odom_frame
        odometry.child_frame_id = args_cli.base_frame
        odometry.pose.pose.position.x = position[0]
        odometry.pose.pose.position.y = position[1]
        odometry.pose.pose.position.z = position[2]
        # Isaac Lab 3.0 and ROS both expose quaternions in XYZW order.
        odometry.pose.pose.orientation.x = quaternion[0]
        odometry.pose.pose.orientation.y = quaternion[1]
        odometry.pose.pose.orientation.z = quaternion[2]
        odometry.pose.pose.orientation.w = quaternion[3]
        odometry.twist.twist.linear.x = linear_velocity[0]
        odometry.twist.twist.linear.y = linear_velocity[1]
        odometry.twist.twist.linear.z = linear_velocity[2]
        odometry.twist.twist.angular.x = angular_velocity[0]
        odometry.twist.twist.angular.y = angular_velocity[1]
        odometry.twist.twist.angular.z = angular_velocity[2]
        self._odom_publisher.publish(odometry)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = args_cli.odom_frame
        transform.child_frame_id = args_cli.base_frame
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation = odometry.pose.pose.orientation
        self._tf.sendTransform(transform)


def _find_base_prim() -> str:
    stage = omni.usd.get_context().get_stage()
    expected = "/World/envs/env_0/Robot/base"
    if stage.GetPrimAtPath(expected).IsValid():
        return expected
    robot_root = "/World/envs/env_0/Robot"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(robot_root) and prim.GetName().lower() == "base":
            return path
    raise RuntimeError(f"could not find a base prim below {robot_root}")


def _laser_scan_metadata(prim) -> dict[str, float | list[float]]:
    """Read the rotary scan geometry required by the ROS LaserScan writer."""

    rotation_rate = float(
        prim.GetAttribute("omni:sensor:Core:scanRateBaseHz").Get() or 0.0
    )
    firing_rate = int(
        prim.GetAttribute("omni:sensor:Core:patternFiringRateHz").Get() or 0
    )
    if rotation_rate <= 0.0 or firing_rate <= 0:
        raise RuntimeError(
            "RTX lidar has invalid scanRateBaseHz or patternFiringRateHz"
        )
    near_range = float(prim.GetAttribute("omni:sensor:Core:nearRangeM").Get() or 0.0)
    far_range = float(prim.GetAttribute("omni:sensor:Core:farRangeM").Get() or 0.0)
    return {
        "horizontalFov": 360.0,
        "horizontalResolution": 360.0 * rotation_rate / firing_rate,
        "depthRange": [near_range, far_range],
        "rotationRate": rotation_rate,
        "azimuthRange": [-180.0, 180.0],
    }


def create_planar_lidar(base_prim: str) -> LidarSensor:
    """Create the legacy rotary control sensor for explicit A/B tests."""

    lidar = Lidar.create(
        path=f"{base_prim}/base_scan",
        config="Example_Rotary_2D",
        tick_rate=10.0,
        translations=[[0.0, 0.0, args_cli.lidar_height]],
    )
    sensor = LidarSensor(lidar, annotators=[])
    sensor.attach_writer(
        "RtxLidarROS2PublishLaserScan",
        topicName=args_cli.scan_topic.lstrip("/"),
        frameId=args_cli.scan_frame,
        **_laser_scan_metadata(lidar.prims[0]),
    )
    return sensor


def _multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two WXYZ quaternions without changing frame conventions."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _mounted_sensor_world_pose(
    robot_data,
    mount_xyz: tuple[float, float, float] | list[float],
    mount_wxyz: tuple[float, float, float, float] | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a base pose with a fixed sensor mount pose."""

    base_position = (
        robot_data.root_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
    )
    base_xyzw = (
        robot_data.root_quat_w.torch[0].detach().cpu().numpy().astype(np.float64)
    )
    base_wxyz = base_xyzw[[3, 0, 1, 2]]
    mount_quaternion = np.asarray(mount_wxyz, dtype=np.float64)
    mount_translation = np.asarray(mount_xyz, dtype=np.float64)
    pure_mount = np.asarray((0.0, *mount_translation), dtype=np.float64)
    base_conjugate = base_wxyz * np.asarray((1.0, -1.0, -1.0, -1.0))
    rotated_mount = _multiply_wxyz(
        _multiply_wxyz(base_wxyz, pure_mount), base_conjugate
    )[1:]
    world_position = base_position + rotated_mount
    world_orientation = _multiply_wxyz(base_wxyz, mount_quaternion)
    return world_position[None, :], world_orientation[None, :]


def _camera_world_pose(robot_data) -> tuple[np.ndarray, np.ndarray]:
    """Compose the robot pose with the calibrated camera mount pose."""

    return _mounted_sensor_world_pose(
        robot_data, args_cli.camera_mount_xyz, camera_usd_wxyz
    )


def create_camera(robot_data) -> CameraSensor | None:
    """Create a calibrated Isaac Sim 6 camera and ROS image writers."""

    if camera_calibration is None:
        return None

    print(
        "[INFO] creating calibrated RTX camera: "
        f"{camera_calibration.width}x{camera_calibration.height}, "
        f"{camera_calibration.lens_model}",
        flush=True,
    )
    attributes = camera_calibration.lens_attributes()
    attributes[f"{camera_calibration.lens_prefix}:imageSize"] = Gf.Vec2i(
        camera_calibration.width, camera_calibration.height
    )
    positions, orientations = _camera_world_pose(robot_data)
    # The audited TinyMal USD is instanceable.  USD Camera prims cannot be
    # authored below an instance proxy, so keep the camera in a global sensor
    # scope and explicitly follow the robot pose every policy tick.
    camera = RtxCamera(
        path="/World/ActuateXSensors/front_camera",
        tick_rate=args_cli.camera_rate,
        positions=positions,
        orientations=orientations,
    )
    print("[INFO] RTX camera prim created", flush=True)
    camera_prim = camera.prims[0]
    camera_prim.ApplyAPI(camera_calibration.lens_schema)
    for name, value in attributes.items():
        camera_prim.GetAttribute(name).Set(value)
    camera_prim.GetAttribute("omni:lensdistortion:model").Set(
        camera_calibration.lens_model
    )
    camera.camera.set_clipping_ranges(
        near_distances=[args_cli.camera_near], far_distances=[args_cli.camera_far]
    )
    camera.camera.set_focus_distances([args_cli.camera_focus_distance])
    camera.camera.set_fstops([args_cli.camera_f_stop])
    camera_prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(
        args_cli.camera_exposure_time
    )
    camera_prim.CreateAttribute("exposure:responsivity", Sdf.ValueTypeNames.Float).Set(
        args_cli.camera_responsivity
    )
    camera_prim.CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(
        args_cli.camera_f_stop
    )
    print("[INFO] RTX camera optics configured", flush=True)

    sensor = CameraSensor(
        camera,
        resolution=(camera_calibration.height, camera_calibration.width),
        annotators=[],
    )
    print("[INFO] RTX camera render product created", flush=True)
    rgb_render_var = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.Rgb.name
    )
    sensor.attach_writer(
        f"{rgb_render_var}ROS2PublishImage",
        topicName=args_cli.camera_topic.lstrip("/"),
        frameId=args_cli.camera_frame,
        queueSize=1,
    )
    print("[INFO] RTX RGB ROS writer attached", flush=True)
    sensor.attach_writer(
        "ROS2PublishCameraInfo",
        topicName=args_cli.camera_info_topic.lstrip("/"),
        frameId=args_cli.camera_frame,
        queueSize=1,
        width=camera_calibration.width,
        height=camera_calibration.height,
        projectionType=camera_calibration.distortion_model,
        k=np.asarray(camera_calibration.k, dtype=np.float64).reshape(1, 9),
        r=np.asarray(camera_calibration.r, dtype=np.float64).reshape(1, 9),
        p=np.asarray(camera_calibration.p, dtype=np.float64).reshape(1, 12),
        physicalDistortionModel=camera_calibration.distortion_model,
        physicalDistortionCoefficients=np.asarray(
            camera_calibration.d, dtype=np.float32
        ),
    )
    print("[INFO] RTX CameraInfo ROS writer attached", flush=True)
    if args_cli.camera_depth_topic:
        depth_render_var = (
            omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                sd.SensorType.DistanceToImagePlane.name
            )
        )
        sensor.attach_writer(
            f"{depth_render_var}ROS2PublishImage",
            topicName=args_cli.camera_depth_topic.lstrip("/"),
            frameId=args_cli.camera_frame,
            queueSize=1,
        )
        print("[INFO] RTX depth ROS writer attached", flush=True)
    return sensor


def update_camera_pose(sensor: CameraSensor | None, robot_data) -> None:
    """Keep a globally authored camera rigidly aligned to the instanceable robot."""

    if sensor is None:
        return
    positions, orientations = _camera_world_pose(robot_data)
    sensor.authoring_object.set_world_poses(positions, orientations)


def set_policy_command(env, observation, command) -> None:
    target = torch.tensor(command, device=env.unwrapped.device)
    env.unwrapped.command_manager.get_command("base_velocity")[:] = target
    observation["policy"][:, 9:12] = target * torch.tensor(
        OBSERVATION_COMMAND_SCALE, device=env.unwrapped.device
    )


def main() -> None:
    env = None
    node = None
    lidar_sensor = None
    mid360_runtime: Mid360Runtime | None = None
    mid360_publisher: Mid360RosPublisher | None = None
    mid360_mount_wxyz: tuple[float, float, float, float] | None = None
    camera_sensor = None
    try:
        # This must be visible while the Isaac Lab environment constructs its
        # SimulationContext.  Setting it after gym.make() leaves the environment
        # with a stale ``has_rtx_sensors`` flag and skips sensor-aware rerenders.
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)
        cfg = TinymalNavigationEnvCfg()
        cfg.seed = 1
        env = gym.make("Isaac-Velocity-Native-Omni-TinyMal-v0", cfg=cfg)
        observation, _ = env.reset()
        actor = load_actor(args_cli.actor, env.unwrapped.device)
        base_prim = _find_base_prim()
        robot_data = env.unwrapped.scene["robot"].data

        rclpy.init(args=None)
        node = Nav2PolicyBridge()
        if args_cli.lidar_type == "mid360":
            mid360_publisher = Mid360RosPublisher(
                node,
                frame_id=args_cli.scan_frame,
                pointcloud_topic=args_cli.pointcloud_topic,
                scan_topic=args_cli.scan_topic,
                custom_topic=args_cli.livox_custom_topic,
                scan_bins=args_cli.scan_bins,
                scan_min_z=args_cli.scan_min_z,
                scan_max_z=args_cli.scan_max_z,
                range_min=0.1,
                range_max=40.0,
            )
            visual_usd = None
            if args_cli.mid360_visual:
                visual_usd = (
                    PROJECT_ROOT
                    / "robots"
                    / "sensors"
                    / "mid360"
                    / "usd"
                    / "mid360_visual.usda"
                )
            mid360_mount_wxyz = _rpy_deg_to_wxyz(args_cli.mid360_mount_rpy_deg)
            mid360_positions, mid360_orientations = _mounted_sensor_world_pose(
                robot_data,
                args_cli.mid360_mount_xyz,
                mid360_mount_wxyz,
            )
            mid360_runtime = create_mid360(
                pattern=mid360_pattern,
                publisher=mid360_publisher,
                position=tuple(mid360_positions[0]),
                orientation_wxyz=tuple(mid360_orientations[0]),
                visual_usd=visual_usd,
                stride=args_cli.mid360_stride,
                motion_compensated=(
                    args_cli.mid360_motion_compensation == "compensated"
                ),
                angular_error_std_deg=args_cli.mid360_angular_error_std_deg,
                range_accuracy_m=args_cli.mid360_range_accuracy_m,
            )
        else:
            lidar_sensor = create_planar_lidar(base_prim)
        camera_sensor = create_camera(robot_data)
        _pump_kit_for_rtx()

        policy_period_ns = round(env.unwrapped.step_dt * 1.0e9)
        sim_time_ns = 0
        step = 0
        reset_count = 0
        set_policy_command(env, observation, (0.0, 0.0, 0.0))
        node.publish_clock()
        node.publish_robot_state(env.unwrapped.scene["robot"].data)
        print(
            "[INFO] TinyMal Nav2 bridge ready: "
            f"{base_prim}, policy_dt={env.unwrapped.step_dt:.3f}s, "
            f"lidar={args_cli.lidar_type}, "
            f"camera={'enabled' if camera_sensor is not None else 'disabled'}",
            flush=True,
        )

        with torch.no_grad():
            while simulation_app.is_running():
                node.set_sim_time(sim_time_ns)
                rclpy.spin_once(node, timeout_sec=0.0)
                command = node.command()
                set_policy_command(env, observation, command)
                actions = actor(observation["policy"])
                observation, _, terminated, truncated, _ = env.step(actions)

                if mid360_runtime is not None:
                    mid360_positions, mid360_orientations = _mounted_sensor_world_pose(
                        env.unwrapped.scene["robot"].data,
                        args_cli.mid360_mount_xyz,
                        mid360_mount_wxyz,
                    )
                    update_mid360_pose(
                        mid360_runtime,
                        mid360_positions,
                        mid360_orientations,
                    )
                update_camera_pose(camera_sensor, env.unwrapped.scene["robot"].data)
                _pump_kit_for_rtx()

                sim_time_ns += policy_period_ns
                step += 1
                node.set_sim_time(sim_time_ns)
                set_policy_command(env, observation, command)
                node.publish_clock()
                node.publish_robot_state(env.unwrapped.scene["robot"].data)

                reset_now = bool(torch.logical_or(terminated, truncated).any().item())
                if reset_now:
                    reset_count += int(
                        torch.logical_or(terminated, truncated).sum().item()
                    )
                    node.get_logger().error("locomotion termination detected")
                    if not args_cli.continue_after_fall:
                        break
                if args_cli.max_steps and step >= args_cli.max_steps:
                    break

        print(
            f"[INFO] navigation rollout stopped after {step} steps; "
            f"terminations={reset_count}",
            flush=True,
        )
        if mid360_publisher is not None and mid360_runtime is not None:
            stats = mid360_publisher.stats
            writer = mid360_runtime.writer
            print(
                "[INFO] Mid-360 output: "
                f"frames={stats.frames}, points={stats.points}, "
                f"last_frame_points={writer.last_point_count}, "
                f"empty_render_frames={writer.empty_frames}, "
                f"offset_ns={writer.last_offset_min_ns}.."
                f"{writer.last_offset_max_ns}, "
                f"shards={writer.last_shards_received}/"
                f"{len(mid360_runtime.sensors)}, "
                f"lines={writer.last_line_histogram.tolist()}, "
                f"custom_msg={stats.custom_msg_enabled}",
                flush=True,
            )
    finally:
        if camera_sensor is not None:
            rgb_render_var = (
                omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                    sd.SensorType.Rgb.name
                )
            )
            camera_sensor.detach_writer(f"{rgb_render_var}ROS2PublishImage")
            camera_sensor.detach_writer("ROS2PublishCameraInfo")
            if args_cli.camera_depth_topic:
                depth_render_var = (
                    omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                        sd.SensorType.DistanceToImagePlane.name
                    )
                )
                camera_sensor.detach_writer(f"{depth_render_var}ROS2PublishImage")
        if mid360_runtime is not None:
            mid360_runtime.writer.detach()
        if lidar_sensor is not None:
            lidar_sensor.detach_writer("RtxLidarROS2PublishLaserScan")
        # Release the render products and experimental prim wrappers while the
        # stage still exists.  Keeping them alive until env.close() makes the
        # Sim 6 prim-deletion callback receive the stage event object instead
        # of a path during shutdown.
        camera_sensor = None
        lidar_sensor = None
        mid360_runtime = None
        gc.collect()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
