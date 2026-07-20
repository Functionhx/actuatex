"""ROS-independent Livox point layout and 3D-to-2D projection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Matches livox_ros_driver2's packed LivoxPointXyzrtlt and PointCloud2 fields.
LIVOX_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
        ("tag", "u1"),
        ("line", "u1"),
        ("timestamp", "<f8"),
    ],
    align=False,
)
LIVOX_POINT_STEP = 26

# sensor_msgs/msg/PointField numeric constants, kept here so this module can
# be tested without a ROS installation.
POINT_FIELDS = (
    ("x", 0, 7),
    ("y", 4, 7),
    ("z", 8, 7),
    ("intensity", 12, 7),
    ("tag", 16, 2),
    ("line", 17, 2),
    ("timestamp", 18, 8),
)


@dataclass(frozen=True, slots=True)
class LaserScanProjection:
    """A projected planar scan and its exact angular metadata."""

    ranges: np.ndarray
    angle_min: float
    angle_max: float
    angle_increment: float


def _one_dimensional(
    value: np.ndarray, name: str, length: int, dtype: np.dtype
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    return array


def build_livox_points(
    xyz: np.ndarray,
    intensity: np.ndarray,
    timebase_ns: int,
    offset_time_ns: np.ndarray,
    line: np.ndarray,
    tag: np.ndarray | None = None,
) -> np.ndarray:
    """Build the driver's packed PointXYZRTLT representation.

    The float64 ``timestamp`` field intentionally follows the official ROS
    driver and stores absolute nanoseconds.  Exact relative nanoseconds remain
    available through Livox ``CustomMsg.offset_time``.
    """

    coordinates = np.asarray(xyz, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {coordinates.shape}")
    point_count = int(coordinates.shape[0])
    reflectivity = _one_dimensional(
        intensity, "intensity", point_count, np.dtype(np.float32)
    )
    offsets = _one_dimensional(
        offset_time_ns, "offset_time_ns", point_count, np.dtype(np.int64)
    )
    lines = _one_dimensional(line, "line", point_count, np.dtype(np.uint8))
    if tag is None:
        tags = np.zeros(point_count, dtype=np.uint8)
    else:
        tags = _one_dimensional(tag, "tag", point_count, np.dtype(np.uint8))
    if np.any(offsets < 0):
        raise ValueError("offset_time_ns must be non-negative")

    points = np.empty(point_count, dtype=LIVOX_POINT_DTYPE)
    points["x"] = coordinates[:, 0]
    points["y"] = coordinates[:, 1]
    points["z"] = coordinates[:, 2]
    points["intensity"] = np.clip(reflectivity, 0.0, 255.0)
    points["tag"] = tags
    points["line"] = lines
    points["timestamp"] = np.asarray(timebase_ns, dtype=np.float64) + offsets
    return points


def project_planar_scan(
    xyz: np.ndarray,
    *,
    bins: int = 1440,
    min_z: float = -0.08,
    max_z: float = 0.08,
    range_min: float = 0.1,
    range_max: float = 40.0,
) -> LaserScanProjection:
    """Project a height band from the 3D cloud into a Nav2 LaserScan."""

    if bins < 8:
        raise ValueError("bins must be at least eight")
    if min_z >= max_z:
        raise ValueError("min_z must be smaller than max_z")
    if range_min <= 0.0 or range_max <= range_min:
        raise ValueError("range limits must satisfy 0 < min < max")

    coordinates = np.asarray(xyz, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {coordinates.shape}")
    planar_range = np.hypot(coordinates[:, 0], coordinates[:, 1])
    valid = np.isfinite(coordinates).all(axis=1)
    valid &= coordinates[:, 2] >= min_z
    valid &= coordinates[:, 2] <= max_z
    valid &= planar_range >= range_min
    valid &= planar_range <= range_max

    angle_min = -float(np.pi)
    angle_increment = 2.0 * float(np.pi) / bins
    angle_max = angle_min + (bins - 1) * angle_increment
    ranges = np.full(bins, np.inf, dtype=np.float32)
    if np.any(valid):
        angles = np.arctan2(coordinates[valid, 1], coordinates[valid, 0])
        indices = np.floor((angles - angle_min) / angle_increment).astype(np.int64)
        indices = np.clip(indices, 0, bins - 1)
        np.minimum.at(ranges, indices, planar_range[valid])
    ranges.setflags(write=False)
    return LaserScanProjection(
        ranges=ranges,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=angle_increment,
    )


if LIVOX_POINT_DTYPE.itemsize != LIVOX_POINT_STEP:
    raise RuntimeError(
        f"Livox packed point must be {LIVOX_POINT_STEP} bytes, "
        f"got {LIVOX_POINT_DTYPE.itemsize}"
    )
