"""Load the pinned Livox Mid-360 non-repetitive firing pattern.

The bundled table is the four-second, 800,000-point pattern published by
Livox in ``livox_laser_simulation``.  Its first column is a one-based point
index despite the historical ``Time/s`` header.  Physical timing therefore
comes from the Mid-360's documented 200 kpoint/s rate, not from that label.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import io
import lzma
from pathlib import Path

import numpy as np


SAMPLE_RATE_HZ = 200_000
FRAME_RATE_HZ = 10
POINT_PERIOD_NS = 1_000_000_000 // SAMPLE_RATE_HZ
POINTS_PER_FRAME = SAMPLE_RATE_HZ // FRAME_RATE_HZ
FRAME_PERIOD_NS = 1_000_000_000 // FRAME_RATE_HZ
CHANNEL_COUNT = 4
POINT_COUNT = 800_000
FRAME_COUNT = POINT_COUNT // POINTS_PER_FRAME
PATTERN_CYCLE_NS = POINT_COUNT * POINT_PERIOD_NS

SOURCE_REPOSITORY = "https://github.com/Livox-SDK/livox_laser_simulation"
SOURCE_COMMIT = "1cce1073633a062b92e30243a4c2920e45551bb5"
SOURCE_PATH = "scan_mode/mid360.csv"
SOURCE_CSV_SHA256 = "aa1fc08b6a4400608dbd6ee832b7ea3a9c3c37197e734f60f58fe5abf762269a"
SOURCE_XZ_SHA256 = "35df3bc5be073ba4dfd8081ac836ddd67adde99af71079578a019839b19a9e5b"


@dataclass(frozen=True, slots=True)
class Mid360Frame:
    """One 10 Hz emitter state in acquisition order."""

    index: int
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    fire_time_ns: np.ndarray
    channel_id: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.azimuth_deg.size)


@dataclass(frozen=True, slots=True)
class Mid360Pattern:
    """The complete four-second scan cycle in Isaac/REP-103 angles."""

    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.azimuth_deg.shape != (POINT_COUNT,):
            raise ValueError(
                f"expected {POINT_COUNT} azimuth samples, got {self.azimuth_deg.shape}"
            )
        if self.elevation_deg.shape != (POINT_COUNT,):
            raise ValueError(
                "expected "
                f"{POINT_COUNT} elevation samples, got {self.elevation_deg.shape}"
            )
        if not np.isfinite(self.azimuth_deg).all():
            raise ValueError("Mid-360 pattern contains non-finite azimuth values")
        if not np.isfinite(self.elevation_deg).all():
            raise ValueError("Mid-360 pattern contains non-finite elevation values")
        if float(self.azimuth_deg.min()) < -180.0:
            raise ValueError("Mid-360 azimuth falls below -180 degrees")
        if float(self.azimuth_deg.max()) >= 180.0:
            raise ValueError("Mid-360 azimuth reaches or exceeds 180 degrees")
        # The official pattern extends about 0.22 degrees beyond the nominal
        # -7..52 degree FOV, consistent with the specified angular uncertainty.
        if float(self.elevation_deg.min()) < -7.3:
            raise ValueError("Mid-360 elevation falls below the official pattern")
        if float(self.elevation_deg.max()) > 52.3:
            raise ValueError("Mid-360 elevation exceeds the official pattern")

    @property
    def point_count(self) -> int:
        return POINT_COUNT

    @property
    def frame_count(self) -> int:
        return FRAME_COUNT

    @property
    def cycle_seconds(self) -> float:
        return PATTERN_CYCLE_NS / 1_000_000_000

    def frame(self, index: int, *, stride: int = 1) -> Mid360Frame:
        """Return one frame, optionally with an explicit acquisition stride."""

        if not 0 <= index < FRAME_COUNT:
            raise IndexError(f"frame index must be in [0, {FRAME_COUNT}), got {index}")
        if stride < 1:
            raise ValueError("stride must be at least one")

        start = index * POINTS_PER_FRAME
        stop = start + POINTS_PER_FRAME
        sample_index = np.arange(start, stop, stride, dtype=np.uint32)
        fire_time_ns = (
            np.arange(0, POINTS_PER_FRAME, stride, dtype=np.uint32) * POINT_PERIOD_NS
        )
        channel_id = np.remainder(sample_index, CHANNEL_COUNT).astype(
            np.uint32, copy=False
        )
        for array in (fire_time_ns, channel_id):
            array.setflags(write=False)
        return Mid360Frame(
            index=index,
            azimuth_deg=self.azimuth_deg[start:stop:stride],
            elevation_deg=self.elevation_deg[start:stop:stride],
            fire_time_ns=fire_time_ns,
            channel_id=channel_id,
        )


def _default_resource():
    return resources.files(__package__).joinpath("data", "mid360.csv.xz")


def _read_payload(path: str | Path | None) -> tuple[bytes, str, bool]:
    if path is None:
        resource = _default_resource()
        return resource.read_bytes(), str(resource), True
    source_path = Path(path).expanduser().resolve()
    return source_path.read_bytes(), str(source_path), False


def load_mid360_pattern(
    path: str | Path | None = None, *, verify_source: bool = True
) -> Mid360Pattern:
    """Load and validate the official pattern or a compatible local table.

    ``.xz`` inputs are decompressed automatically.  The bundled table is
    checked against both its compressed and original CSV SHA-256 digests.
    Custom tables must retain the original three-column layout and the exact
    800,000-row, one-based acquisition index.
    """

    payload, source, is_default = _read_payload(path)
    if is_default and verify_source:
        compressed_digest = hashlib.sha256(payload).hexdigest()
        if compressed_digest != SOURCE_XZ_SHA256:
            raise ValueError(
                "bundled Mid-360 compressed pattern checksum mismatch: "
                f"{compressed_digest}"
            )
    if source.endswith(".xz"):
        csv_payload = lzma.decompress(payload)
    else:
        csv_payload = payload
    if is_default and verify_source:
        csv_digest = hashlib.sha256(csv_payload).hexdigest()
        if csv_digest != SOURCE_CSV_SHA256:
            raise ValueError(f"bundled Mid-360 CSV checksum mismatch: {csv_digest}")

    values = np.loadtxt(
        io.BytesIO(csv_payload),
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )
    if values.shape != (POINT_COUNT, 3):
        raise ValueError(
            f"Mid-360 table must have shape ({POINT_COUNT}, 3), got {values.shape}"
        )
    point_index = values[:, 0]
    expected_index = np.arange(1, POINT_COUNT + 1, dtype=np.float64)
    if not np.array_equal(point_index, expected_index):
        raise ValueError("Mid-360 table acquisition index is not contiguous")

    # Livox stores azimuth in [0, 360] and zenith down from +Z.  Isaac RTX and
    # REP-103 use azimuth [-180, 180), elevation up from the XY plane.
    azimuth_deg = np.remainder(values[:, 1] + 180.0, 360.0) - 180.0
    elevation_deg = 90.0 - values[:, 2]
    azimuth_deg = np.ascontiguousarray(azimuth_deg, dtype=np.float32)
    elevation_deg = np.ascontiguousarray(elevation_deg, dtype=np.float32)
    azimuth_deg.setflags(write=False)
    elevation_deg.setflags(write=False)
    return Mid360Pattern(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        source=source,
    )
