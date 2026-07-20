"""Simulator-independent locomotion and deployment building blocks.

Isaac Lab, MuJoCo and a hardware runtime must agree on more than an ONNX
tensor shape.  This module owns the small, testable pieces that commonly drift:
joint permutations, observation-history layout, action latency and the outer
safety envelope.  It intentionally depends only on NumPy so the same code can
be used by tests, MuJoCo evaluation and a deployment reference implementation.

The safety helpers are a research guard rail, not a certified functional-safety
system.  A real robot still needs vendor limits, an independent emergency stop
and a hardware-level state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _names(names: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(str(name) for name in names)
    if not result:
        raise ValueError(f"{label} cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate joint names")
    return result


def _last_axis(values: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0 or result.shape[-1] != size:
        raise ValueError(
            f"{label} must have last dimension {size}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result


def joint_permutation(
    source_names: Sequence[str], target_names: Sequence[str]
) -> np.ndarray:
    """Return indices that reorder a source joint vector into target order."""

    source = _names(source_names, "source_names")
    target = _names(target_names, "target_names")
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        extra = sorted(set(source) - set(target))
        raise ValueError(
            "joint sets differ: "
            f"missing_from_source={missing}, extra_in_source={extra}"
        )
    source_index = {name: index for index, name in enumerate(source)}
    return np.asarray([source_index[name] for name in target], dtype=np.int64)


def reorder_joints(
    values: np.ndarray,
    source_names: Sequence[str],
    target_names: Sequence[str],
) -> np.ndarray:
    """Reorder any array whose last axis is a named joint axis."""

    source = _names(source_names, "source_names")
    values = _last_axis(values, len(source), "values")
    return values[..., joint_permutation(source, target_names)]


class TermMajorHistory:
    """Rolling observation history matching Isaac Lab's term-major layout.

    Each term stores samples from oldest to newest.  Flattening concatenates all
    history for term 0, then all history for term 1, and so on.  This differs
    from concatenating complete frames and is an important deployment detail.
    Leading batch dimensions are preserved, so this class also works for a
    vectorized MuJoCo environment.
    """

    def __init__(
        self,
        term_dimensions: Mapping[str, int],
        history_length: int,
    ) -> None:
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if not term_dimensions:
            raise ValueError("term_dimensions cannot be empty")
        self.term_dimensions = {
            str(name): int(size) for name, size in term_dimensions.items()
        }
        if any(size <= 0 for size in self.term_dimensions.values()):
            raise ValueError("every observation term dimension must be positive")
        self.history_length = int(history_length)
        self._buffers: dict[str, np.ndarray] | None = None

    @property
    def observation_dim(self) -> int:
        return self.history_length * sum(self.term_dimensions.values())

    def _validated_terms(
        self, terms: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if set(terms) != set(self.term_dimensions):
            missing = sorted(set(self.term_dimensions) - set(terms))
            extra = sorted(set(terms) - set(self.term_dimensions))
            raise ValueError(f"observation terms differ: missing={missing}, extra={extra}")
        result = {
            name: _last_axis(terms[name], size, name)
            for name, size in self.term_dimensions.items()
        }
        leading_shapes = {value.shape[:-1] for value in result.values()}
        if len(leading_shapes) != 1:
            raise ValueError("all observation terms must share leading dimensions")
        return result

    def reset(self, terms: Mapping[str, np.ndarray]) -> np.ndarray:
        """Fill every history slot with the current observation."""

        values = self._validated_terms(terms)
        self._buffers = {
            name: np.repeat(value[..., np.newaxis, :], self.history_length, axis=-2)
            for name, value in values.items()
        }
        return self.flatten()

    def append(self, terms: Mapping[str, np.ndarray]) -> np.ndarray:
        """Append one sample and return the flattened term-major observation."""

        if self._buffers is None:
            return self.reset(terms)
        values = self._validated_terms(terms)
        for name, value in values.items():
            buffer = self._buffers[name]
            if buffer.shape[:-2] != value.shape[:-1]:
                raise ValueError(
                    f"leading shape for {name} changed from "
                    f"{buffer.shape[:-2]} to {value.shape[:-1]}"
                )
            buffer[..., :-1, :] = buffer[..., 1:, :]
            buffer[..., -1, :] = value
        return self.flatten()

    def reset_mask(
        self,
        mask: np.ndarray,
        terms: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Reset selected members of a vectorized history buffer."""

        if self._buffers is None:
            return self.reset(terms)
        values = self._validated_terms(terms)
        mask = np.asarray(mask, dtype=bool)
        leading_shape = next(iter(values.values())).shape[:-1]
        if mask.shape != leading_shape:
            raise ValueError(f"mask must have shape {leading_shape}, got {mask.shape}")
        for name, value in values.items():
            repeated = np.repeat(
                value[..., np.newaxis, :], self.history_length, axis=-2
            )
            self._buffers[name][mask] = repeated[mask]
        return self.flatten()

    def flatten(self) -> np.ndarray:
        if self._buffers is None:
            raise RuntimeError("history has not been initialized")
        flattened = []
        for name in self.term_dimensions:
            buffer = self._buffers[name]
            flattened.append(
                buffer.reshape(buffer.shape[:-2] + (-1,))
            )
        result = np.concatenate(flattened, axis=-1)
        if result.shape[-1] != self.observation_dim:
            raise AssertionError("internal observation-history shape drifted")
        return result.astype(np.float32, copy=False)


class ActionDelayLine:
    """Fixed-step action latency for simulation randomization and deployment."""

    def __init__(self, action_dim: int, delay_steps: int) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if delay_steps < 0:
            raise ValueError("delay_steps cannot be negative")
        self.action_dim = int(action_dim)
        self.delay_steps = int(delay_steps)
        self._queue: np.ndarray | None = None

    def reset(self, action: np.ndarray) -> np.ndarray:
        action = _last_axis(action, self.action_dim, "action")
        if self.delay_steps == 0:
            self._queue = None
        else:
            self._queue = np.repeat(
                action[..., np.newaxis, :], self.delay_steps, axis=-2
            )
        return action.copy()

    def push(self, action: np.ndarray) -> np.ndarray:
        action = _last_axis(action, self.action_dim, "action")
        if self.delay_steps == 0:
            return action.copy()
        if self._queue is None:
            self.reset(np.zeros_like(action))
        assert self._queue is not None
        if self._queue.shape[:-2] != action.shape[:-1]:
            raise ValueError("action leading shape changed after delay-line reset")
        delayed = self._queue[..., 0, :].copy()
        self._queue[..., :-1, :] = self._queue[..., 1:, :]
        self._queue[..., -1, :] = action
        return delayed


@dataclass(frozen=True)
class SafetyDecision:
    enabled: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SafetyEnvelope:
    """Outer policy-runtime limits shared by simulation and deployment tests."""

    command_lower: tuple[float, float, float]
    command_upper: tuple[float, float, float]
    command_acceleration: tuple[float, float, float]
    action_limit: float = 1.0
    maximum_tilt_rad: float = 0.8
    watchdog_timeout_s: float = 0.10

    def __post_init__(self) -> None:
        lower = np.asarray(self.command_lower, dtype=np.float64)
        upper = np.asarray(self.command_upper, dtype=np.float64)
        acceleration = np.asarray(self.command_acceleration, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or acceleration.shape != (3,):
            raise ValueError("command limits must contain vx, vy and yaw rate")
        if np.any(lower >= upper):
            raise ValueError("each command lower bound must be below its upper bound")
        if np.any(acceleration <= 0.0):
            raise ValueError("command acceleration limits must be positive")
        if self.action_limit <= 0.0:
            raise ValueError("action_limit must be positive")
        if not 0.0 < self.maximum_tilt_rad < np.pi:
            raise ValueError("maximum_tilt_rad must be in (0, pi)")
        if self.watchdog_timeout_s <= 0.0:
            raise ValueError("watchdog_timeout_s must be positive")

    def slew_command(
        self,
        requested: np.ndarray,
        previous: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        requested = _last_axis(requested, 3, "requested command")
        previous = _last_axis(previous, 3, "previous command")
        if requested.shape != previous.shape:
            raise ValueError("requested and previous commands must have equal shape")
        lower = np.asarray(self.command_lower, dtype=np.float64)
        upper = np.asarray(self.command_upper, dtype=np.float64)
        acceleration = np.asarray(self.command_acceleration, dtype=np.float64)
        requested = np.clip(requested, lower, upper)
        maximum_delta = acceleration * dt
        return previous + np.clip(requested - previous, -maximum_delta, maximum_delta)

    def evaluate(
        self,
        projected_gravity_body: np.ndarray,
        state_age_s: float,
        joint_position: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
    ) -> SafetyDecision:
        reasons: list[str] = []
        try:
            gravity = _last_axis(projected_gravity_body, 3, "projected gravity")
            joint_position = np.asarray(joint_position, dtype=np.float64)
            joint_lower = np.asarray(joint_lower, dtype=np.float64)
            joint_upper = np.asarray(joint_upper, dtype=np.float64)
            if joint_position.shape != joint_lower.shape or joint_position.shape != joint_upper.shape:
                reasons.append("joint_shape_mismatch")
            elif not (
                np.isfinite(joint_position).all()
                and np.isfinite(joint_lower).all()
                and np.isfinite(joint_upper).all()
            ):
                reasons.append("non_finite_joint_state")
            elif np.any((joint_position < joint_lower) | (joint_position > joint_upper)):
                reasons.append("joint_limit")

            gravity_norm = np.linalg.norm(gravity, axis=-1)
            if np.any(gravity_norm < 1.0e-6):
                reasons.append("invalid_gravity")
            else:
                cosine = np.clip(-gravity[..., 2] / gravity_norm, -1.0, 1.0)
                if np.any(np.arccos(cosine) > self.maximum_tilt_rad):
                    reasons.append("tilt")
        except ValueError:
            reasons.append("non_finite_state")

        if not np.isfinite(state_age_s) or state_age_s > self.watchdog_timeout_s:
            reasons.append("watchdog")
        return SafetyDecision(enabled=not reasons, reasons=tuple(dict.fromkeys(reasons)))

    def sanitize_action(self, action: np.ndarray, enabled: bool = True) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if not np.isfinite(action).all() or not enabled:
            return np.zeros_like(action, dtype=np.float64)
        return np.clip(action, -self.action_limit, self.action_limit)
