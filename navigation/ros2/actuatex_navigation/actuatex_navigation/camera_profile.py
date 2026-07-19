"""Validated camera calibration and frame conversions for simulation.

The loader accepts the YAML shape emitted by ``camera_calibration`` and by a
ROS 2 ``sensor_msgs/CameraInfo`` dump.  It deliberately contains no Isaac Sim
or ROS imports so calibration mistakes can be caught before launching Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


_PINHOLE_COEFFICIENTS = (
    "k1",
    "k2",
    "p1",
    "p2",
    "k3",
    "k4",
    "k5",
    "k6",
    "s1",
    "s2",
    "s3",
    "s4",
)
_FISHEYE_COEFFICIENTS = ("k1", "k2", "k3", "k4")

# A forward-facing camera on a REP-103 base.  Isaac/USD camera coordinates are
# +Y up / -Z forward.  ROS optical coordinates are +X right / +Y down / +Z
# forward.  Values here are WXYZ; ROS messages are converted to XYZW at the API
# boundary.
_BASE_TO_USD_CAMERA_WXYZ = (0.5, 0.5, -0.5, -0.5)
_BASE_TO_ROS_OPTICAL_WXYZ = (0.5, -0.5, 0.5, -0.5)


def _sequence(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        value = value.get("data")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence or a mapping with 'data'")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _matrix(
    payload: Mapping[str, Any],
    key: str,
    length: int,
    default: tuple[float, ...] | None = None,
    aliases: tuple[str, ...] = (),
) -> tuple[float, ...]:
    selected_key = next(
        (candidate for candidate in (key, *aliases) if candidate in payload), key
    )
    value = payload.get(selected_key)
    if value is None:
        if default is None:
            alternatives = "/".join((key, *aliases))
            raise ValueError(
                f"missing required camera calibration field: {alternatives}"
            )
        return default
    result = _sequence(value, selected_key)
    if length >= 0 and len(result) != length:
        raise ValueError(
            f"{selected_key} must contain {length} values, got {len(result)}"
        )
    return result


@dataclass(frozen=True)
class CameraCalibration:
    """A single calibrated ROS camera model."""

    width: int
    height: int
    distortion_model: str
    k: tuple[float, ...]
    d: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]
    camera_name: str = "camera"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CameraCalibration":
        """Create and validate a profile from a ROS-style mapping."""

        width = int(payload.get("image_width", payload.get("width", 0)))
        height = int(payload.get("image_height", payload.get("height", 0)))
        if width <= 0 or height <= 0:
            raise ValueError("image_width and image_height must be positive")

        k = _matrix(payload, "camera_matrix", 9, aliases=("k",))
        if k[0] <= 0.0 or k[4] <= 0.0:
            raise ValueError("camera focal lengths fx and fy must be positive")
        if not math.isclose(k[8], 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("camera_matrix[8] must equal 1")
        if any(abs(k[index]) > 1.0e-9 for index in (1, 3, 6, 7)):
            raise ValueError(
                "Isaac Sim OpenCV cameras do not support skewed K matrices"
            )

        distortion_model = str(payload.get("distortion_model", "")).strip()
        if distortion_model not in {
            "plumb_bob",
            "rational_polynomial",
            "equidistant",
        }:
            raise ValueError(
                "distortion_model must be plumb_bob, rational_polynomial, or equidistant"
            )
        d = _matrix(payload, "distortion_coefficients", -1, aliases=("d",))
        expected_lengths = {
            "plumb_bob": {5},
            "rational_polynomial": {8, 12},
            "equidistant": {4},
        }[distortion_model]
        if len(d) not in expected_lengths:
            expected = "/".join(str(length) for length in sorted(expected_lengths))
            raise ValueError(
                f"{distortion_model} requires {expected} distortion coefficients, got {len(d)}"
            )

        identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        r = _matrix(
            payload,
            "rectification_matrix",
            9,
            default=identity,
            aliases=("r",),
        )
        default_p = (
            k[0],
            k[1],
            k[2],
            0.0,
            k[3],
            k[4],
            k[5],
            0.0,
            k[6],
            k[7],
            k[8],
            0.0,
        )
        p = _matrix(
            payload,
            "projection_matrix",
            12,
            default=default_p,
            aliases=("p",),
        )
        camera_name = str(payload.get("camera_name", "camera")).strip() or "camera"
        return cls(width, height, distortion_model, k, d, r, p, camera_name)

    @property
    def lens_schema(self) -> str:
        if self.distortion_model == "equidistant":
            return "OmniLensDistortionOpenCvFisheyeAPI"
        return "OmniLensDistortionOpenCvPinholeAPI"

    @property
    def lens_model(self) -> str:
        return (
            "opencvFisheye"
            if self.distortion_model == "equidistant"
            else "opencvPinhole"
        )

    def lens_attributes(self) -> dict[str, float]:
        """Return Isaac Sim 6 lens-schema attributes except ``imageSize``."""

        if self.distortion_model == "equidistant":
            prefix = "omni:lensdistortion:opencvFisheye"
            names = _FISHEYE_COEFFICIENTS
            coefficients = self.d
        else:
            prefix = "omni:lensdistortion:opencvPinhole"
            names = _PINHOLE_COEFFICIENTS
            coefficients = self.d + (0.0,) * (len(names) - len(self.d))
        attributes = {
            f"{prefix}:fx": self.k[0],
            f"{prefix}:fy": self.k[4],
            f"{prefix}:cx": self.k[2],
            f"{prefix}:cy": self.k[5],
        }
        attributes.update(
            {f"{prefix}:{name}": value for name, value in zip(names, coefficients)}
        )
        return attributes

    @property
    def lens_prefix(self) -> str:
        if self.distortion_model == "equidistant":
            return "omni:lensdistortion:opencvFisheye"
        return "omni:lensdistortion:opencvPinhole"


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    """Load a camera calibration YAML without silently accepting bad fields."""

    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    with calibration_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"camera calibration root must be a mapping: {calibration_path}"
        )
    return CameraCalibration.from_mapping(payload)


def _normalize_wxyz(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
    if norm <= 1.0e-12:
        raise ValueError("zero-length quaternion")
    return tuple(float(value) / norm for value in quaternion)  # type: ignore[return-value]


def _multiply_wxyz(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rpy_wxyz(rpy_degrees: Sequence[float]) -> tuple[float, float, float, float]:
    if len(rpy_degrees) != 3:
        raise ValueError("camera mount RPY must contain roll, pitch, and yaw")
    roll, pitch, yaw = (math.radians(float(value)) for value in rpy_degrees)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return _normalize_wxyz(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def camera_frame_quaternions(
    mount_rpy_degrees: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return Isaac USD WXYZ and ROS optical XYZW mount quaternions.

    ``mount_rpy_degrees`` follows URDF/REP-103 fixed-axis roll, pitch, yaw for a
    nominal camera whose forward direction is the robot's +X axis.
    """

    mount = _rpy_wxyz(mount_rpy_degrees)
    usd_wxyz = _normalize_wxyz(_multiply_wxyz(mount, _BASE_TO_USD_CAMERA_WXYZ))
    optical_wxyz = _normalize_wxyz(_multiply_wxyz(mount, _BASE_TO_ROS_OPTICAL_WXYZ))
    ros_xyzw = (
        optical_wxyz[1],
        optical_wxyz[2],
        optical_wxyz[3],
        optical_wxyz[0],
    )
    return usd_wxyz, ros_xyzw


def rotate_vector_wxyz(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate a vector; exposed to make frame-convention tests explicit."""

    if len(vector) != 3:
        raise ValueError("vector must contain three values")
    q = _normalize_wxyz(quaternion)
    pure = (0.0, float(vector[0]), float(vector[1]), float(vector[2]))
    conjugate = (q[0], -q[1], -q[2], -q[3])
    rotated = _multiply_wxyz(_multiply_wxyz(q, pure), conjugate)
    return rotated[1], rotated[2], rotated[3]
