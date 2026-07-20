"""Livox Mid-360 RTX lidar integration for Isaac Sim 6.0.1.

The sensor uses Livox's published 800,000-ray, four-second non-repetitive
pattern as forty solid-state emitter states.  Each state is one physical
10 Hz frame with 20,000 individually timed rays and four hardware channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import omni
import omni.replicator.core as rep
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.sensors.experimental.rtx import (
    Lidar,
    LidarSensor,
    parse_generic_model_output_data,
)
from omni.replicator.core import Writer
from pxr import Vt

from actuatex_navigation.mid360_pattern import (
    CHANNEL_COUNT,
    FRAME_COUNT,
    FRAME_RATE_HZ,
    Mid360Pattern,
)


MID360_WRITER_NAME = "ActuateXMid360Writer"
# This keeps forty states and their five per-emitter arrays below RTX Hydra's
# fixed 5 MiB serialized-attribute ceiling (about 3.82 MiB per shard).
MAX_EMITTERS_PER_RTX_PRIM = 5_000
_WRITER_REGISTERED = False


@dataclass(frozen=True, slots=True)
class Mid360Runtime:
    """Objects that must remain alive for one RTX Mid-360 instance."""

    sensors: tuple[LidarSensor, ...]
    writer: Writer
    lidars: tuple[Lidar, ...]
    visual_mount: XformPrim | None
    lidar_paths: tuple[str, ...]


def _float_array(values: np.ndarray) -> Vt.FloatArray:
    return Vt.FloatArray.FromNumpy(np.ascontiguousarray(values, dtype=np.float32))


def _uint_array(values: np.ndarray) -> Vt.UIntArray:
    return Vt.UIntArray.FromNumpy(np.ascontiguousarray(values, dtype=np.uint32))


def build_mid360_attributes(
    pattern: Mid360Pattern,
    *,
    stride: int = 1,
    emitter_indices: np.ndarray | None = None,
    motion_compensated: bool = False,
    angular_error_std_deg: float = 0.05,
    range_accuracy_m: float = 0.02,
) -> dict[str, Any]:
    """Translate the official Livox pattern into Isaac RTX USD attributes.

    ``stride=1`` is the fidelity path: 200,000 fired rays/s.  Larger strides
    are an explicit performance/debug approximation while retaining the
    original per-ray timestamps for the rays that remain.
    """

    if stride < 1:
        raise ValueError("Mid-360 pattern stride must be at least one")
    if angular_error_std_deg < 0.0:
        raise ValueError("angular error standard deviation cannot be negative")
    if range_accuracy_m < 0.0:
        raise ValueError("range accuracy cannot be negative")
    if pattern.frame_count != FRAME_COUNT:
        raise ValueError(
            f"Mid-360 pattern must contain {FRAME_COUNT} frames, "
            f"got {pattern.frame_count}"
        )

    first_frame = pattern.frame(0, stride=stride)
    if emitter_indices is None:
        emitter_indices = np.arange(first_frame.point_count, dtype=np.int64)
    else:
        emitter_indices = np.asarray(emitter_indices, dtype=np.int64).reshape(-1)
        if emitter_indices.size == 0:
            raise ValueError("an RTX emitter shard cannot be empty")
        if emitter_indices[0] < 0 or emitter_indices[-1] >= first_frame.point_count:
            raise ValueError("emitter shard index is outside the retained pattern")
        if np.any(np.diff(emitter_indices) <= 0):
            raise ValueError("emitter shard indices must be strictly increasing")
    shard_point_count = int(emitter_indices.size)
    rays_per_line = np.bincount(
        first_frame.channel_id[emitter_indices], minlength=CHANNEL_COUNT
    ).astype(np.uint32, copy=False)
    attributes: dict[str, Any] = {
        "omni:sensor:marketName": "Mid-360",
        "omni:sensor:modelName": "Mid-360",
        "omni:sensor:modelVendor": "Livox",
        "omni:sensor:modelVersion": "official-pattern-2021",
        "omni:sensor:Core:scanType": "SOLID_STATE",
        "omni:sensor:Core:nearRangeM": 0.1,
        "omni:sensor:Core:farRangeM": 70.0,
        "omni:sensor:Core:minReflectance": 0.1,
        "omni:sensor:Core:minReflectionRangeM": 40.0,
        "omni:sensor:Core:rangeResolutionM": 0.001,
        "omni:sensor:Core:rangeAccuracyM": float(range_accuracy_m),
        "omni:sensor:Core:azimuthErrorMean": 0.0,
        "omni:sensor:Core:azimuthErrorStd": float(angular_error_std_deg),
        "omni:sensor:Core:elevationErrorMean": 0.0,
        "omni:sensor:Core:elevationErrorStd": float(angular_error_std_deg),
        "omni:sensor:Core:maxReturns": 1,
        "omni:sensor:Core:scanRateBaseHz": FRAME_RATE_HZ,
        "omni:sensor:Core:patternFiringRateHz": FRAME_RATE_HZ,
        "omni:sensor:Core:numberOfEmitters": shard_point_count,
        # RTX ``channelId`` identifies a detector, not Livox's 0..3 packet
        # ``line``.  Collapsing all sequential firings onto four RTX detector
        # IDs suppresses returns in the Core model.  Give each emitter a stable
        # internal detector ID and retain the four physical Livox lines in
        # ``bank``; the ROS writer reconstructs line from emitterId.
        "omni:sensor:Core:numberOfChannels": shard_point_count,
        "omni:sensor:Core:stateResolutionStep": 1,
        "omni:sensor:Core:numLines": CHANNEL_COUNT,
        "omni:sensor:Core:numRaysPerLine": _uint_array(rays_per_line),
        "omni:sensor:Core:elementsCoordsType": "CARTESIAN",
        "omni:sensor:Core:outputFrameOfReference": "SENSOR",
        "omni:sensor:Core:outputMotionCompensationState": (
            "COMPENSATED" if motion_compensated else "NONCOMPENSATED"
        ),
        "omni:sensor:Core:instantLidar": False,
        "omni:sensor:Core:intensityProcessing": "NORMALIZATION",
        "omni:sensor:Core:intensityScalePercent": 255.0,
        "omni:sensor:Core:waveLengthNm": 905.0,
        "omni:sensor:Core:rayType": "IDEALIZED",
        "omni:sensor:Core:skipDroppingInvalidPoints": False,
    }

    for state_index in range(FRAME_COUNT):
        frame = pattern.frame(state_index, stride=stride)
        if frame.point_count != first_frame.point_count:
            raise ValueError("all Mid-360 emitter states must have equal length")
        prefix = f"omni:sensor:Core:emitterState:s{state_index + 1:03d}"
        attributes[f"{prefix}:azimuthDeg"] = _float_array(
            frame.azimuth_deg[emitter_indices]
        )
        attributes[f"{prefix}:elevationDeg"] = _float_array(
            frame.elevation_deg[emitter_indices]
        )
        attributes[f"{prefix}:fireTimeNs"] = _uint_array(
            frame.fire_time_ns[emitter_indices]
        )
        attributes[f"{prefix}:channelId"] = _uint_array(
            np.arange(1, shard_point_count + 1, dtype=np.uint32)
        )
        attributes[f"{prefix}:bank"] = _uint_array(frame.channel_id[emitter_indices])
    return attributes


def split_mid360_emitters(point_count: int) -> tuple[np.ndarray, ...]:
    """Split one frame below RTX Hydra's fixed 5 MiB attribute limit.

    Five arrays are authored for each of forty emitter states.  At full Livox
    density one prim serializes to about 15.3 MiB, while Isaac Sim 6.0.1 caps a
    sensor at 5 MiB.  Interleaved shards preserve every firing and timestamp;
    synchronized outputs are merged by the writer below.  Do not infer a 1024
    global channel limit from the bundled test asset: Sim 6 accepts the 5000
    one-based detector IDs used here, while excessive sensor prims make the
    multi-render-product FIFO less reliable.
    """

    if point_count < 1:
        raise ValueError("Mid-360 frame must contain at least one emitter")
    shard_count = (point_count + MAX_EMITTERS_PER_RTX_PRIM - 1) // (
        MAX_EMITTERS_PER_RTX_PRIM
    )
    all_indices = np.arange(point_count, dtype=np.int64)
    return tuple(all_indices[index::shard_count] for index in range(shard_count))


def author_mid360_visual(
    *,
    position: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    visual_usd: Path | None,
) -> XformPrim | None:
    """Create the optional teaching shell independently of the RTX sensor."""

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World/ActuateXSensors", "Xform")
    if visual_usd is None:
        return None

    visual_path = visual_usd.expanduser().resolve()
    if not visual_path.is_file():
        raise FileNotFoundError(f"Mid-360 visual USD not found: {visual_path}")
    visual_prim_path = "/World/ActuateXSensors/mid360_visual"
    visual_prim = stage.DefinePrim(visual_prim_path, "Xform")
    visual_prim.GetReferences().AddReference(str(visual_path))
    return XformPrim(
        visual_prim_path,
        positions=np.asarray([position], dtype=np.float64),
        orientations=np.asarray([orientation_wxyz], dtype=np.float64),
        reset_xform_op_properties=True,
    )


def _gmo_array(value: Any, count: int, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.size < count:
        raise RuntimeError(
            f"GenericModelOutput array has {array.size} values; expected {count}"
        )
    return array[:count]


class ActuateXMid360Writer(Writer):
    """Convert one RTX GenericModelOutput frame into Livox ROS messages."""

    version = "1.0.0"

    def __init__(
        self,
        publisher: Any = None,
        line_by_lidar_path: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.version = "1.0.0"
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput")]
        self._publisher = publisher
        self.frames_received = 0
        self.empty_frames = 0
        self.last_point_count = 0
        self.last_offset_min_ns: int | None = None
        self.last_offset_max_ns: int | None = None
        self.last_line_histogram = np.zeros(CHANNEL_COUNT, dtype=np.int64)
        self.last_shards_received = 0
        self._line_by_lidar_path: dict[str, np.ndarray] = {}
        for lidar_path, line_by_emitter in (line_by_lidar_path or {}).items():
            lines = np.asarray(line_by_emitter, dtype=np.uint8).reshape(-1)
            if lines.size == 0:
                raise ValueError("line_by_emitter cannot be empty")
            if np.any(lines >= CHANNEL_COUNT):
                raise ValueError("Livox line IDs must be in [0, 3]")
            self._line_by_lidar_path[str(lidar_path)] = lines.copy()

    def write(self, data: dict[str, Any]) -> None:
        if self._publisher is None or "renderProducts" not in data:
            return
        shard_xyz: list[np.ndarray] = []
        shard_intensity: list[np.ndarray] = []
        shard_offsets: list[np.ndarray] = []
        shard_lines: list[np.ndarray] = []
        shard_timestamps: list[int] = []
        for _render_product, render_data in data["renderProducts"].items():
            gmo_raw = render_data.get("GenericModelOutput")
            if isinstance(gmo_raw, dict):
                gmo_raw = gmo_raw.get("data")
            if gmo_raw is None:
                continue
            gmo = parse_generic_model_output_data(gmo_raw)
            point_count = int(gmo.numElements)
            if point_count <= 0:
                continue

            x = _gmo_array(gmo.x, point_count, np.dtype(np.float32))
            y = _gmo_array(gmo.y, point_count, np.dtype(np.float32))
            z = _gmo_array(gmo.z, point_count, np.dtype(np.float32))
            xyz = np.column_stack((x, y, z)).astype(np.float32, copy=False)
            intensity = _gmo_array(gmo.scalar, point_count, np.dtype(np.float32))
            offsets = _gmo_array(gmo.timeOffsetNs, point_count, np.dtype(np.int64))
            emitter_ids = _gmo_array(gmo.emitterId, point_count, np.dtype(np.uint32))
            lidar_path = str(render_data.get("camera", ""))
            line_by_emitter = self._line_by_lidar_path.get(lidar_path)
            if line_by_emitter is None and len(self._line_by_lidar_path) == 1:
                line_by_emitter = next(iter(self._line_by_lidar_path.values()))
            if line_by_emitter is None:
                raise RuntimeError(
                    f"Mid-360 writer has no emitter mapping for {lidar_path!r}"
                )
            if np.any(emitter_ids >= line_by_emitter.size):
                raise RuntimeError(
                    "RTX returned an emitter ID outside the Mid-360 state"
                )
            lines = line_by_emitter[emitter_ids]

            finite = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
            if not np.all(finite):
                xyz = xyz[finite]
                intensity = intensity[finite]
                offsets = offsets[finite]
                lines = lines[finite]
            shard_xyz.append(xyz)
            shard_intensity.append(intensity)
            shard_offsets.append(offsets)
            shard_lines.append(np.asarray(lines, dtype=np.uint8))
            shard_timestamps.append(int(gmo.timestampNs))

        if not shard_xyz:
            self.empty_frames += 1
            self.last_shards_received = 0
            return

        timebase_ns = min(shard_timestamps)
        normalized_offsets = [
            offsets + (timestamp_ns - timebase_ns)
            for offsets, timestamp_ns in zip(shard_offsets, shard_timestamps)
        ]
        xyz = np.concatenate(shard_xyz)
        intensity = np.concatenate(shard_intensity)
        offsets = np.concatenate(normalized_offsets)
        lines = np.concatenate(shard_lines)
        firing_order = np.argsort(offsets, kind="stable")
        xyz = xyz[firing_order]
        intensity = intensity[firing_order]
        offsets = offsets[firing_order]
        lines = lines[firing_order]
        self._publisher.publish(
            timebase_ns=timebase_ns,
            xyz=xyz,
            intensity=intensity,
            offset_time_ns=offsets,
            line=lines,
        )
        self.frames_received += 1
        self.last_shards_received = len(shard_xyz)
        self.last_point_count = int(xyz.shape[0])
        if offsets.size:
            self.last_offset_min_ns = int(offsets.min())
            self.last_offset_max_ns = int(offsets.max())
        self.last_line_histogram = np.bincount(lines, minlength=CHANNEL_COUNT).astype(
            np.int64, copy=False
        )

    def write_metadata(self, *_args: Any, **_kwargs: Any) -> None:
        """This streaming writer intentionally has no Replicator disk backend."""


def _register_writer() -> None:
    global _WRITER_REGISTERED
    if not _WRITER_REGISTERED:
        rep.WriterRegistry.register(ActuateXMid360Writer)
        _WRITER_REGISTERED = True


def create_mid360(
    *,
    pattern: Mid360Pattern,
    publisher: Any,
    position: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    visual_usd: Path | None,
    stride: int = 1,
    motion_compensated: bool = False,
    angular_error_std_deg: float = 0.05,
    range_accuracy_m: float = 0.02,
) -> Mid360Runtime:
    """Author, render, and attach the Livox-compatible ROS writer."""

    print(
        "[INFO] authoring Mid-360 mount and official emitter states: "
        f"stride={stride}, motion_compensated={motion_compensated}",
        flush=True,
    )
    visual_mount = author_mid360_visual(
        position=position,
        orientation_wxyz=orientation_wxyz,
        visual_usd=visual_usd,
    )
    # Keep each RTX prim at a global, unnested path.  In Isaac Lab's
    # Fabric scene delegate, a dynamically-updated parent Xform plus RTX
    # geometry streaming can submit stale transforms and has caused Vulkan
    # DEVICE_LOST on the first sensor render.  Lidar already inherits
    # XformPrim, so it can follow the robot directly without that extra parent.
    retained_frame = pattern.frame(0, stride=stride)
    emitter_shards = split_mid360_emitters(retained_frame.point_count)
    print(
        "[INFO] Mid-360 RTX attributes ready: "
        f"states={FRAME_COUNT}, rays_per_state={retained_frame.point_count}, "
        f"rtx_shards={len(emitter_shards)}",
        flush=True,
    )
    lidars: list[Lidar] = []
    sensors: list[LidarSensor] = []
    line_by_lidar_path: dict[str, np.ndarray] = {}
    for shard_index, emitter_indices in enumerate(emitter_shards):
        if len(emitter_shards) == 1:
            lidar_path = "/World/ActuateXSensors/mid360_lidar"
        else:
            lidar_path = f"/World/ActuateXSensors/mid360_lidar_shard_{shard_index:02d}"
        attributes = build_mid360_attributes(
            pattern,
            stride=stride,
            emitter_indices=emitter_indices,
            motion_compensated=motion_compensated,
            angular_error_std_deg=angular_error_std_deg,
            range_accuracy_m=range_accuracy_m,
        )
        lidar = Lidar(
            lidar_path,
            attributes=attributes,
            tick_rate=float(FRAME_RATE_HZ),
            # Accumulate the multi-render subframes into one physical 10 Hz
            # Livox frame.  Without this, Replicator exposes partial firings at
            # the render cadence instead of one 100 ms point cloud.
            accumulate_outputs=True,
            aux_output_level="BASIC",
            positions=np.asarray([position], dtype=np.float64),
            orientations=np.asarray([orientation_wxyz], dtype=np.float64),
        )
        print(
            f"[INFO] Mid-360 RTX prim created: {lidar_path}, "
            f"emitters={emitter_indices.size}",
            flush=True,
        )
        lidars.append(lidar)
        sensors.append(LidarSensor(lidar, annotators=[]))
        line_by_lidar_path[lidar_path] = retained_frame.channel_id[emitter_indices]
    print(f"[INFO] Mid-360 render products created: {len(sensors)}", flush=True)
    _register_writer()
    writer = rep.writers.get(MID360_WRITER_NAME)
    writer.initialize(
        publisher=publisher,
        line_by_lidar_path=line_by_lidar_path,
    )
    render_product_paths = [
        sensor.render_product.GetPath().pathString for sensor in sensors
    ]
    writer.attach(render_product_paths)
    print("[INFO] Mid-360 Livox ROS writer attached", flush=True)
    return Mid360Runtime(
        sensors=tuple(sensors),
        writer=writer,
        lidars=tuple(lidars),
        visual_mount=visual_mount,
        lidar_paths=tuple(line_by_lidar_path),
    )


def update_mid360_pose(
    runtime: Mid360Runtime,
    positions: np.ndarray,
    orientations_wxyz: np.ndarray,
) -> None:
    """Rigidly follow an instanceable robot without authoring below it."""

    for lidar in runtime.lidars:
        lidar.set_world_poses(positions, orientations_wxyz)
    if runtime.visual_mount is not None:
        runtime.visual_mount.set_world_poses(positions, orientations_wxyz)
