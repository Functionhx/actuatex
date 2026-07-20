from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.robomaster.evaluation import (  # noqa: E402
    SENTINEL_COMMAND_SEGMENTS,
    evaluation_settle_steps,
    ramped_command,
)


def test_shared_protocol_is_the_canonical_twenty_two_seconds() -> None:
    assert sum(segment.duration_s for segment in SENTINEL_COMMAND_SEGMENTS) == 22.0
    assert [segment.name for segment in SENTINEL_COMMAND_SEGMENTS] == [
        "stand",
        "forward_0p5",
        "forward_1p0",
        "backward_0p5",
        "yaw_0p8",
        "arc_0p7_0p6",
    ]


def test_ramp_reaches_target_without_overshoot() -> None:
    start = (0.0, 0.0, 0.0)
    target = (1.0, 0.0, -0.5)
    assert ramped_command(
        start,
        target,
        step=0,
        dt=0.02,
        ramp_duration_s=0.10,
    ) == pytest.approx((0.2, 0.0, -0.1))
    assert ramped_command(
        start,
        target,
        step=20,
        dt=0.02,
        ramp_duration_s=0.10,
    ) == target
    assert ramped_command(
        start,
        target,
        step=0,
        dt=0.02,
        ramp_duration_s=0.0,
    ) == target


def test_settle_window_covers_the_command_ramp() -> None:
    assert evaluation_settle_steps(
        num_steps=200,
        duration_s=4.0,
        dt=0.02,
        ramp_duration_s=0.5,
    ) == 50
    assert evaluation_settle_steps(
        num_steps=10,
        duration_s=0.2,
        dt=0.02,
        ramp_duration_s=1.0,
    ) == 9
