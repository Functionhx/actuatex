"""Simulator-independent velocity limiting for a learned locomotion policy."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VelocityCommand:
    """Planar velocity command in the robot body frame."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @classmethod
    def zero(cls) -> "VelocityCommand":
        return cls()


@dataclass(frozen=True)
class VelocityLimits:
    """Policy envelope and per-axis slew-rate limits."""

    max_forward: float = 0.55
    max_reverse: float = 0.25
    max_lateral: float = 0.25
    max_yaw: float = 0.75
    acceleration_x: float = 0.80
    acceleration_y: float = 0.60
    acceleration_yaw: float = 1.20
    deadband_x: float = 0.02
    deadband_y: float = 0.02
    deadband_yaw: float = 0.04

    def __post_init__(self) -> None:
        positive = (
            self.max_forward,
            self.max_reverse,
            self.max_lateral,
            self.max_yaw,
            self.acceleration_x,
            self.acceleration_y,
            self.acceleration_yaw,
        )
        deadbands = (self.deadband_x, self.deadband_y, self.deadband_yaw)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError(
                "velocity and acceleration limits must be finite and positive"
            )
        if not all(math.isfinite(value) and value >= 0.0 for value in deadbands):
            raise ValueError("deadbands must be finite and non-negative")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _deadband(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def _approach(current: float, target: float, maximum_change: float) -> float:
    return current + _clamp(target - current, -maximum_change, maximum_change)


class CommandFilter:
    """Clamp Nav2 output to the trained policy envelope and smooth transitions."""

    def __init__(self, limits: VelocityLimits | None = None) -> None:
        self.limits = limits or VelocityLimits()
        self._current = VelocityCommand.zero()

    @property
    def current(self) -> VelocityCommand:
        return self._current

    def limit_target(self, target: VelocityCommand) -> VelocityCommand:
        values = (target.x, target.y, target.yaw)
        if not all(math.isfinite(value) for value in values):
            return VelocityCommand.zero()
        return VelocityCommand(
            x=_deadband(
                _clamp(target.x, -self.limits.max_reverse, self.limits.max_forward),
                self.limits.deadband_x,
            ),
            y=_deadband(
                _clamp(target.y, -self.limits.max_lateral, self.limits.max_lateral),
                self.limits.deadband_y,
            ),
            yaw=_deadband(
                _clamp(target.yaw, -self.limits.max_yaw, self.limits.max_yaw),
                self.limits.deadband_yaw,
            ),
        )

    def update(self, target: VelocityCommand, dt: float) -> VelocityCommand:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        target = self.limit_target(target)
        self._current = VelocityCommand(
            x=_approach(self._current.x, target.x, self.limits.acceleration_x * dt),
            y=_approach(self._current.y, target.y, self.limits.acceleration_y * dt),
            yaw=_approach(
                self._current.yaw,
                target.yaw,
                self.limits.acceleration_yaw * dt,
            ),
        )
        return self._current

    def stop(self) -> VelocityCommand:
        """Immediately zero the command for watchdog and shutdown paths."""

        self._current = VelocityCommand.zero()
        return self._current
