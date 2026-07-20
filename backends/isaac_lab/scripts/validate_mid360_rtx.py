#!/usr/bin/env python
"""Validate the ActuateX Mid-360 emitter model in an isolated RTX scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
NAVIGATION_PYTHON_ROOT = PROJECT_ROOT / "navigation" / "ros2" / "actuatex_navigation"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(NAVIGATION_PYTHON_ROOT))

parser = argparse.ArgumentParser(
    description="Run the official Livox pattern inside a closed RTX test scene."
)
parser.add_argument("--stride", type=int, default=101)
parser.add_argument("--frames", type=int, default=180)
parser.add_argument(
    "--out",
    type=Path,
    help="Optional JSON path for machine-readable validation evidence.",
)
parser.add_argument(
    "--channel_layout",
    choices=("rtx", "collapsed_livox_line"),
    default="rtx",
    help="collapsed_livox_line reproduces the invalid four-detector mapping",
)
args = parser.parse_args()
if args.stride < 1 or args.frames < 1:
    parser.error("--stride and --frames must be positive")

from isaacsim import SimulationApp  # noqa: E402


simulation_app = SimulationApp(
    {
        "headless": True,
        "enable_motion_bvh": True,
        "extra_args": [
            "--/rtx/hydra/supportMultiTickRate=true",
            "--/rtx/rendering/perSensorTickTlas=true",
        ],
    }
)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.experimental.objects import Cube  # noqa: E402
from isaacsim.sensors.experimental.rtx import (  # noqa: E402
    Lidar,
    LidarSensor,
    parse_generic_model_output_data,
)
from omni.replicator.core import Writer  # noqa: E402
from pxr import Vt  # noqa: E402

from actuatex_navigation.mid360_pattern import (  # noqa: E402
    FRAME_COUNT,
    load_mid360_pattern,
)
from tinymal_lab.mid360_rtx import (  # noqa: E402
    build_mid360_attributes,
    create_mid360,
    split_mid360_emitters,
)


class Mid360ValidationPublisher:
    """Collect merged physical frames from the production ROS writer path."""

    def __init__(self) -> None:
        self.frames = 0
        self.points = 0
        self.last_point_count = 0
        self.last_offsets = np.empty(0, dtype=np.int64)
        self.last_lines = np.empty(0, dtype=np.uint8)
        self.offset_order_valid = True

    def publish(
        self,
        *,
        timebase_ns: int,
        xyz: np.ndarray,
        intensity: np.ndarray,
        offset_time_ns: np.ndarray,
        line: np.ndarray,
    ) -> None:
        del timebase_ns, intensity
        self.frames += 1
        self.last_point_count = int(xyz.shape[0])
        self.points += self.last_point_count
        self.last_offsets = np.asarray(offset_time_ns, dtype=np.int64).copy()
        self.last_lines = np.asarray(line, dtype=np.uint8).copy()
        self.offset_order_valid &= bool(
            self.last_offsets.size < 2 or np.all(np.diff(self.last_offsets) >= 0)
        )


class Mid360ValidationWriter(Writer):
    """Collect only the fields needed to prove that the RTX profile fires."""

    version = "1.0.0"

    def __init__(self) -> None:
        self.version = "1.0.0"
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput")]
        self.valid_frames = 0
        self.empty_frames = 0
        self.points = 0
        self.last_time_offsets = np.empty(0, dtype=np.int64)
        self.last_channel_ids = np.empty(0, dtype=np.uint32)

    def write(self, data: dict[str, object]) -> None:
        for render_data in data.get("renderProducts", {}).values():
            raw = render_data.get("GenericModelOutput")
            if isinstance(raw, dict):
                raw = raw.get("data")
            if raw is None:
                self.empty_frames += 1
                continue
            gmo = parse_generic_model_output_data(raw)
            count = int(gmo.numElements)
            if count == 0:
                self.empty_frames += 1
                continue
            self.valid_frames += 1
            self.points += count
            self.last_time_offsets = np.asarray(gmo.timeOffsetNs)[:count].copy()
            self.last_channel_ids = np.asarray(gmo.channelId)[:count].copy()

    def write_metadata(self, *_args, **_kwargs) -> None:
        """This diagnostic intentionally has no Replicator disk backend."""


def create_closed_scene() -> None:
    """Surround the optical origin so every valid firing intersects geometry."""

    slabs = {
        "x_pos": ((5.0, 0.0, 0.0), (1.0, 12.0, 12.0)),
        "x_neg": ((-5.0, 0.0, 0.0), (1.0, 12.0, 12.0)),
        "y_pos": ((0.0, 5.0, 0.0), (12.0, 1.0, 12.0)),
        "y_neg": ((0.0, -5.0, 0.0), (12.0, 1.0, 12.0)),
        "z_pos": ((0.0, 0.0, 5.0), (12.0, 12.0, 1.0)),
        "z_neg": ((0.0, 0.0, -5.0), (12.0, 12.0, 1.0)),
    }
    for name, (position, scale) in slabs.items():
        Cube(
            f"/World/targets/{name}",
            positions=np.asarray(position, dtype=np.float64),
            scales=np.asarray(scale, dtype=np.float64),
        )


def write_result(result: dict[str, object]) -> None:
    """Persist a deterministic validation record when requested."""

    if args.out is None:
        return
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[VALIDATE] evidence: {output_path}", flush=True)


def main() -> int:
    create_closed_scene()
    pattern = load_mid360_pattern()
    rays_per_state = pattern.frame(0, stride=args.stride).point_count
    if args.channel_layout == "rtx":
        emitter_shards = split_mid360_emitters(rays_per_state)
        covered_emitters = np.sort(np.concatenate(emitter_shards))
        emitter_coverage_valid = np.array_equal(
            covered_emitters, np.arange(rays_per_state, dtype=np.int64)
        )
        publisher = Mid360ValidationPublisher()
        runtime = create_mid360(
            pattern=pattern,
            publisher=publisher,
            position=(0.0, 0.0, 0.0),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            visual_usd=None,
            stride=args.stride,
        )
        expected_states = {
            f"OmniSensorGenericLidarCoreEmitterStateAPI:s{index + 1:03d}"
            for index in range(FRAME_COUNT)
        }
        authored_emitters = 0
        for lidar in runtime.lidars:
            schemas = set(lidar.prims[0].GetAppliedSchemas())
            missing_states = sorted(expected_states - schemas)
            if missing_states:
                raise RuntimeError(f"missing emitter schemas: {missing_states}")
            authored_emitters += int(
                lidar.prims[0].GetAttribute("omni:sensor:Core:numberOfEmitters").Get()
            )
        print(
            "[VALIDATE] profile: "
            f"stride={args.stride}, rays_per_state={rays_per_state}, "
            f"states={FRAME_COUNT}/{FRAME_COUNT}, "
            f"rtx_shards={len(emitter_shards)}, "
            f"emissions_authored={authored_emitters}/{rays_per_state}, "
            f"coverage={emitter_coverage_valid}, "
            "channel_layout=rtx",
            flush=True,
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(args.frames):
            simulation_app.update()
        timeline.stop()
        writer = runtime.writer
        if publisher.last_offsets.size:
            offset_summary = (
                f"{int(publisher.last_offsets.min())}.."
                f"{int(publisher.last_offsets.max())} ns"
            )
            line_histogram = np.bincount(publisher.last_lines, minlength=4).tolist()
        else:
            offset_summary = "none"
            line_histogram = []
        print(
            "[VALIDATE] output: "
            f"frames={publisher.frames}, points={publisher.points}, "
            f"last_frame_points={publisher.last_point_count}, "
            f"offsets={offset_summary}, lines={line_histogram}, "
            f"shards={writer.last_shards_received}/{len(runtime.sensors)}, "
            f"time_sorted={publisher.offset_order_valid}",
            flush=True,
        )
        runtime.writer.detach()
        complete = (
            publisher.frames > 0
            and writer.last_shards_received == len(runtime.sensors)
            and publisher.offset_order_valid
            and emitter_coverage_valid
            and authored_emitters == rays_per_state
        )
        write_result(
            {
                "backend": "Isaac Sim 6.0.1 GA RTX Lidar",
                "channel_layout": "rtx",
                "stride": args.stride,
                "simulation_updates": args.frames,
                "pattern_states_authored": FRAME_COUNT,
                "pattern_states_expected": FRAME_COUNT,
                "rays_per_state": rays_per_state,
                "emitters_authored": authored_emitters,
                "emitter_coverage_valid": bool(emitter_coverage_valid),
                "rtx_shards": len(emitter_shards),
                "output_frames": publisher.frames,
                "output_points": publisher.points,
                "last_frame_points": publisher.last_point_count,
                "last_offset_min_ns": (
                    int(publisher.last_offsets.min())
                    if publisher.last_offsets.size
                    else None
                ),
                "last_offset_max_ns": (
                    int(publisher.last_offsets.max())
                    if publisher.last_offsets.size
                    else None
                ),
                "last_line_histogram": line_histogram,
                "last_shards_received": writer.last_shards_received,
                "time_sorted": bool(publisher.offset_order_valid),
                "passed": bool(complete),
            }
        )
        return 0 if complete else 2

    # Negative-control path: reproduce the tempting but invalid mapping of all
    # emitters onto only the four Livox packet line numbers.
    attributes = build_mid360_attributes(pattern, stride=args.stride)
    attributes["omni:sensor:Core:numberOfChannels"] = 4
    for state_index in range(FRAME_COUNT):
        prefix = f"omni:sensor:Core:emitterState:s{state_index + 1:03d}"
        line_ids = pattern.frame(state_index, stride=args.stride).channel_id
        attributes[f"{prefix}:channelId"] = Vt.UIntArray.FromNumpy(
            np.ascontiguousarray(line_ids + 1, dtype=np.uint32)
        )

    lidar = Lidar(
        "/World/mid360",
        attributes=attributes,
        tick_rate=10.0,
        accumulate_outputs=True,
        aux_output_level="BASIC",
    )
    schemas = set(lidar.prims[0].GetAppliedSchemas())
    expected_states = {
        f"OmniSensorGenericLidarCoreEmitterStateAPI:s{index + 1:03d}"
        for index in range(FRAME_COUNT)
    }
    missing_states = sorted(expected_states - schemas)
    print(
        "[VALIDATE] profile: "
        f"stride={args.stride}, rays_per_state={rays_per_state}, "
        f"states={len(expected_states - set(missing_states))}/{FRAME_COUNT}, "
        f"channel_layout={args.channel_layout}",
        flush=True,
    )
    if missing_states:
        raise RuntimeError(f"missing emitter schemas: {missing_states}")

    rep.WriterRegistry.register(Mid360ValidationWriter)
    sensor = LidarSensor(lidar, annotators=[])
    writer = sensor.attach_writer("Mid360ValidationWriter")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(args.frames):
        simulation_app.update()
    timeline.stop()

    if writer.last_time_offsets.size:
        offset_summary = (
            f"{int(writer.last_time_offsets.min())}.."
            f"{int(writer.last_time_offsets.max())} ns"
        )
        channels = np.unique(writer.last_channel_ids).tolist()
    else:
        offset_summary = "none"
        channels = []
    print(
        "[VALIDATE] output: "
        f"valid_frames={writer.valid_frames}, empty_frames={writer.empty_frames}, "
        f"points={writer.points}, offsets={offset_summary}, channels={channels[:16]}",
        flush=True,
    )
    sensor.detach_writer("Mid360ValidationWriter")
    complete = writer.valid_frames > 0
    write_result(
        {
            "backend": "Isaac Sim 6.0.1 GA RTX Lidar",
            "channel_layout": args.channel_layout,
            "stride": args.stride,
            "simulation_updates": args.frames,
            "pattern_states_authored": FRAME_COUNT,
            "pattern_states_expected": FRAME_COUNT,
            "rays_per_state": rays_per_state,
            "valid_frames": writer.valid_frames,
            "empty_frames": writer.empty_frames,
            "output_points": writer.points,
            "last_offset_min_ns": (
                int(writer.last_time_offsets.min())
                if writer.last_time_offsets.size
                else None
            ),
            "last_offset_max_ns": (
                int(writer.last_time_offsets.max())
                if writer.last_time_offsets.size
                else None
            ),
            "last_channels": channels,
            "passed": bool(complete),
        }
    )
    return 0 if complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
