"""ROS 2 publishers for Isaac Sim Mid-360 GenericModelOutput frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mid360_pattern import FRAME_PERIOD_NS
from .mid360_pointcloud import (
    LIVOX_POINT_STEP,
    POINT_FIELDS,
    build_livox_points,
    project_planar_scan,
)


@dataclass(frozen=True, slots=True)
class Mid360PublishStats:
    frames: int
    points: int
    last_timebase_ns: int | None
    custom_msg_enabled: bool


def _stamp(message: Any, timestamp_ns: int) -> None:
    message.sec = int(timestamp_ns // 1_000_000_000)
    message.nanosec = int(timestamp_ns % 1_000_000_000)


class Mid360RosPublisher:
    """Publish official-layout PointCloud2, optional CustomMsg, and LaserScan."""

    def __init__(
        self,
        node: Any,
        *,
        frame_id: str,
        pointcloud_topic: str,
        scan_topic: str = "",
        custom_topic: str = "",
        scan_bins: int = 1440,
        scan_min_z: float = -0.08,
        scan_max_z: float = 0.08,
        range_min: float = 0.1,
        range_max: float = 40.0,
    ) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan, PointCloud2, PointField

        self._node = node
        self._frame_id = frame_id
        self._scan_bins = scan_bins
        self._scan_min_z = scan_min_z
        self._scan_max_z = scan_max_z
        self._range_min = range_min
        self._range_max = range_max
        self._pointcloud_type = PointCloud2
        self._point_field_type = PointField
        self._laser_scan_type = LaserScan
        self._pointcloud_publisher = node.create_publisher(
            PointCloud2, pointcloud_topic, qos_profile_sensor_data
        )
        self._scan_publisher = None
        if scan_topic:
            self._scan_publisher = node.create_publisher(
                LaserScan, scan_topic, qos_profile_sensor_data
            )

        self._custom_msg_type = None
        self._custom_point_type = None
        self._custom_publisher = None
        if custom_topic:
            try:
                from livox_ros_driver2.msg import CustomMsg, CustomPoint
            except ImportError:
                node.get_logger().warning(
                    "livox_ros_driver2 messages are unavailable; publishing the "
                    "official PointXYZRTLT PointCloud2 only"
                )
            else:
                self._custom_msg_type = CustomMsg
                self._custom_point_type = CustomPoint
                self._custom_publisher = node.create_publisher(
                    CustomMsg, custom_topic, qos_profile_sensor_data
                )
        self._frames = 0
        self._points = 0
        self._last_timebase_ns: int | None = None

    @property
    def stats(self) -> Mid360PublishStats:
        return Mid360PublishStats(
            frames=self._frames,
            points=self._points,
            last_timebase_ns=self._last_timebase_ns,
            custom_msg_enabled=self._custom_publisher is not None,
        )

    def _ordered(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray,
        offset_time_ns: np.ndarray,
        line: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        offsets = np.asarray(offset_time_ns, dtype=np.int64)
        if offsets.ndim != 1:
            raise ValueError("offset_time_ns must be one-dimensional")
        if offsets.size > 1 and np.any(offsets[1:] < offsets[:-1]):
            order = np.argsort(offsets, kind="stable")
            return xyz[order], intensity[order], offsets[order], line[order]
        return xyz, intensity, offsets, line

    def publish(
        self,
        *,
        timebase_ns: int,
        xyz: np.ndarray,
        intensity: np.ndarray,
        offset_time_ns: np.ndarray,
        line: np.ndarray,
    ) -> None:
        """Publish one acquisition-ordered 10 Hz sensor frame."""

        xyz, intensity, offsets, line = self._ordered(
            np.asarray(xyz, dtype=np.float32),
            np.asarray(intensity, dtype=np.float32),
            offset_time_ns,
            np.asarray(line, dtype=np.uint8),
        )
        points = build_livox_points(
            xyz=xyz,
            intensity=intensity,
            timebase_ns=timebase_ns,
            offset_time_ns=offsets,
            line=line,
        )

        cloud = self._pointcloud_type()
        cloud.header.frame_id = self._frame_id
        _stamp(cloud.header.stamp, timebase_ns)
        cloud.height = 1
        cloud.width = int(points.size)
        cloud.fields = [
            self._point_field_type(
                name=name,
                offset=offset,
                datatype=datatype,
                count=1,
            )
            for name, offset, datatype in POINT_FIELDS
        ]
        cloud.is_bigendian = False
        cloud.point_step = LIVOX_POINT_STEP
        cloud.row_step = cloud.width * cloud.point_step
        cloud.is_dense = True
        cloud.data = points.tobytes(order="C")
        self._pointcloud_publisher.publish(cloud)

        if self._scan_publisher is not None:
            projection = project_planar_scan(
                xyz,
                bins=self._scan_bins,
                min_z=self._scan_min_z,
                max_z=self._scan_max_z,
                range_min=self._range_min,
                range_max=self._range_max,
            )
            scan = self._laser_scan_type()
            scan.header = cloud.header
            scan.angle_min = projection.angle_min
            scan.angle_max = projection.angle_max
            scan.angle_increment = projection.angle_increment
            scan.time_increment = 0.0
            scan.scan_time = FRAME_PERIOD_NS / 1_000_000_000
            scan.range_min = self._range_min
            scan.range_max = self._range_max
            scan.ranges = projection.ranges.tolist()
            self._scan_publisher.publish(scan)

        if self._custom_publisher is not None:
            self._publish_custom(cloud, timebase_ns, xyz, intensity, offsets, line)

        self._frames += 1
        self._points += int(points.size)
        self._last_timebase_ns = int(timebase_ns)

    def _publish_custom(
        self,
        cloud: Any,
        timebase_ns: int,
        xyz: np.ndarray,
        intensity: np.ndarray,
        offsets: np.ndarray,
        line: np.ndarray,
    ) -> None:
        custom = self._custom_msg_type()
        custom.header = cloud.header
        custom.timebase = int(timebase_ns)
        custom.point_num = int(xyz.shape[0])
        custom.lidar_id = 0
        custom.rsvd = [0, 0, 0]
        reflectivity = np.clip(intensity, 0.0, 255.0).astype(np.uint8)
        points = []
        for index in range(xyz.shape[0]):
            point = self._custom_point_type()
            point.offset_time = int(offsets[index])
            point.x = float(xyz[index, 0])
            point.y = float(xyz[index, 1])
            point.z = float(xyz[index, 2])
            point.reflectivity = int(reflectivity[index])
            point.tag = 0
            point.line = int(line[index])
            points.append(point)
        custom.points = points
        self._custom_publisher.publish(custom)
