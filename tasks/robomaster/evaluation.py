"""Backend-neutral command protocol for Sentinel controller evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CommandSegment:
    name: str
    duration_s: float
    command: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.name or not math.isfinite(self.duration_s):
            raise ValueError("command segment name and duration are invalid")
        if self.duration_s <= 0.0 or len(self.command) != 3:
            raise ValueError("command segment must have positive duration and 3-D command")
        if not all(math.isfinite(value) for value in self.command):
            raise ValueError("command segment contains a non-finite value")


SENTINEL_COMMAND_SEGMENTS = (
    CommandSegment("stand", 2.0, (0.0, 0.0, 0.0)),
    CommandSegment("forward_0p5", 4.0, (0.5, 0.0, 0.0)),
    CommandSegment("forward_1p0", 4.0, (1.0, 0.0, 0.0)),
    CommandSegment("backward_0p5", 4.0, (-0.5, 0.0, 0.0)),
    CommandSegment("yaw_0p8", 4.0, (0.0, 0.0, 0.8)),
    CommandSegment("arc_0p7_0p6", 4.0, (0.7, 0.0, 0.6)),
)


def ramped_command(
    start: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    step: int,
    dt: float,
    ramp_duration_s: float,
) -> tuple[float, float, float]:
    """Interpolate a command, reaching the target after ``ramp_duration_s``."""

    if step < 0 or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("step and dt must be non-negative and positive")
    if not math.isfinite(ramp_duration_s) or ramp_duration_s < 0.0:
        raise ValueError("ramp duration must be finite and non-negative")
    alpha = (
        1.0
        if ramp_duration_s == 0.0
        else min(1.0, (step + 1) * dt / ramp_duration_s)
    )
    return tuple(
        initial + alpha * (final - initial)
        for initial, final in zip(start, target, strict=True)
    )


def evaluation_settle_steps(
    *,
    num_steps: int,
    duration_s: float,
    dt: float,
    ramp_duration_s: float,
) -> int:
    if num_steps <= 0 or min(duration_s, dt) <= 0.0:
        raise ValueError("evaluation duration and step count must be positive")
    natural_settle = round(min(1.0, duration_s / 3.0) / dt)
    ramp_settle = math.ceil(ramp_duration_s / dt)
    return min(num_steps - 1, max(natural_settle, ramp_settle))
