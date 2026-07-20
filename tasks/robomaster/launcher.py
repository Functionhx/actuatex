"""Friction-wheel launcher, feeder, heat gate, recoil and ballistics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

import numpy as np

from .contract import RMUC_2026


class LauncherPhase(str, Enum):
    DISARMED = "disarmed"
    SPINUP = "spinup"
    READY = "ready"
    FEEDING = "feeding"
    COOLDOWN = "cooldown"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class LauncherConfig:
    projectile_mass_kg: float = RMUC_2026.projectile_mass_kg
    projectile_radius_m: float = 0.5 * RMUC_2026.projectile_diameter_m
    maximum_launch_speed_mps: float = RMUC_2026.initial_launch_speed_limit_mps
    target_launch_speed_mps: float = 24.0
    flywheel_radius_m: float = 0.030
    flywheel_inertia_kg_m2: float = 8.1e-5
    flywheel_viscous_friction: float = 2.0e-4
    wheel_to_projectile_efficiency: float = 0.88
    maximum_flywheel_torque_nm: float = 0.45
    feeder_inertia_kg_m2: float = 8.3e-5
    feeder_viscous_friction: float = 0.020
    feeder_stiffness: float = 3.0
    feeder_damping: float = 0.060
    maximum_feeder_torque_nm: float = 1.5
    feeder_pockets: int = 8
    feeder_position_tolerance_rad: float = 0.025
    flywheel_speed_tolerance_ratio: float = 0.025
    maximum_gimbal_error_rad: float = math.radians(1.5)
    cooldown_s: float = 0.080
    barrel_heat_limit: float = RMUC_2026.full_auto_barrel_heat_limit
    barrel_heat_cooling_per_second: float = RMUC_2026.full_auto_cooling_per_second
    barrel_heat_per_projectile: float = RMUC_2026.barrel_heat_per_projectile

    @property
    def target_flywheel_speed_rad_s(self) -> float:
        return self.target_launch_speed_mps / (
            self.flywheel_radius_m * self.wheel_to_projectile_efficiency
        )

    @property
    def feeder_step_rad(self) -> float:
        return 2.0 * math.pi / self.feeder_pockets


@dataclass(frozen=True)
class LauncherState:
    phase: LauncherPhase = LauncherPhase.DISARMED
    left_flywheel_velocity: float = 0.0
    right_flywheel_velocity: float = 0.0
    feeder_position: float = 0.0
    feeder_velocity: float = 0.0
    feeder_target: float = 0.0
    barrel_heat: float = 0.0
    ammunition: int = 0
    cooldown_remaining_s: float = 0.0
    fault_reason: str = ""


@dataclass(frozen=True)
class FireRequest:
    trigger: bool
    armed: bool
    referee_enabled: bool
    safety_clear: bool
    gimbal_error_rad: float
    requested_launch_speed_mps: float


@dataclass(frozen=True)
class ProjectileLaunch:
    speed_mps: float
    kinetic_energy_j: float
    recoil_impulse_body_ns: np.ndarray
    barrel_heat_after: float
    ammunition_after: int


@dataclass(frozen=True)
class LauncherStepResult:
    state: LauncherState
    left_flywheel_torque_nm: float
    right_flywheel_torque_nm: float
    feeder_torque_nm: float
    launch: ProjectileLaunch | None
    rejection_reasons: tuple[str, ...]


def _fire_rejection_reasons(
    state: LauncherState,
    request: FireRequest,
    config: LauncherConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not request.armed:
        reasons.append("disarmed")
    if not request.referee_enabled:
        reasons.append("referee_disabled")
    if not request.safety_clear:
        reasons.append("unsafe_line_of_fire")
    if abs(request.gimbal_error_rad) > config.maximum_gimbal_error_rad:
        reasons.append("gimbal_error")
    if not 0.0 < request.requested_launch_speed_mps <= config.maximum_launch_speed_mps:
        reasons.append("launch_speed_limit")
    if state.ammunition <= 0:
        reasons.append("empty")
    if state.barrel_heat + config.barrel_heat_per_projectile > config.barrel_heat_limit:
        reasons.append("barrel_heat")
    if state.cooldown_remaining_s > 0.0:
        reasons.append("cooldown")
    target = request.requested_launch_speed_mps / (
        config.flywheel_radius_m * config.wheel_to_projectile_efficiency
    )
    tolerance = config.flywheel_speed_tolerance_ratio * target
    if abs(state.left_flywheel_velocity - target) > tolerance:
        reasons.append("left_flywheel_speed")
    if abs(state.right_flywheel_velocity + target) > tolerance:
        reasons.append("right_flywheel_speed")
    return tuple(reasons)


class FrictionWheelLauncher:
    """Deterministic L2 mechanism with explicit flywheel and feeder states."""

    def __init__(self, config: LauncherConfig = LauncherConfig()) -> None:
        self.config = config
        self.state = LauncherState()
        self._pending_request: FireRequest | None = None

    def reset(self, ammunition: int = 300) -> LauncherState:
        if ammunition < 0:
            raise ValueError("ammunition cannot be negative")
        self.state = LauncherState(ammunition=int(ammunition))
        self._pending_request = None
        return self.state

    def safe_stop(self, reason: str) -> LauncherState:
        self._pending_request = None
        self.state = replace(
            self.state,
            phase=LauncherPhase.SAFE_STOP,
            fault_reason=str(reason),
        )
        return self.state

    def step(
        self,
        request: FireRequest,
        dt: float,
    ) -> LauncherStepResult:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        config = self.config
        state = self.state
        cooled_heat = max(
            0.0,
            state.barrel_heat - config.barrel_heat_cooling_per_second * dt,
        )
        cooldown = max(0.0, state.cooldown_remaining_s - dt)

        if state.phase == LauncherPhase.SAFE_STOP:
            self.state = replace(
                state,
                left_flywheel_velocity=self._integrate_flywheel(
                    state.left_flywheel_velocity, 0.0, dt
                ),
                right_flywheel_velocity=self._integrate_flywheel(
                    state.right_flywheel_velocity, 0.0, dt
                ),
                barrel_heat=cooled_heat,
                cooldown_remaining_s=cooldown,
            )
            return LauncherStepResult(self.state, 0.0, 0.0, 0.0, None, (state.fault_reason,))

        target_speed = (
            request.requested_launch_speed_mps
            / (config.flywheel_radius_m * config.wheel_to_projectile_efficiency)
            if request.armed
            else 0.0
        )
        left_torque = self._flywheel_velocity_torque(
            target_speed, state.left_flywheel_velocity
        )
        right_torque = self._flywheel_velocity_torque(
            -target_speed, state.right_flywheel_velocity
        )
        left_velocity = self._integrate_flywheel(
            state.left_flywheel_velocity, left_torque, dt
        )
        right_velocity = self._integrate_flywheel(
            state.right_flywheel_velocity, right_torque, dt
        )

        phase = state.phase
        feeder_target = state.feeder_target
        pending = self._pending_request
        rejection_reasons: tuple[str, ...] = ()
        if not request.armed:
            phase = LauncherPhase.DISARMED
            pending = None
        elif phase == LauncherPhase.DISARMED:
            phase = LauncherPhase.SPINUP

        provisional = replace(
            state,
            left_flywheel_velocity=left_velocity,
            right_flywheel_velocity=right_velocity,
            barrel_heat=cooled_heat,
            cooldown_remaining_s=cooldown,
            phase=phase,
        )
        rejection_reasons = _fire_rejection_reasons(provisional, request, config)
        flywheels_ready = not any(
            reason in rejection_reasons
            for reason in ("left_flywheel_speed", "right_flywheel_speed")
        )
        if phase == LauncherPhase.SPINUP and flywheels_ready:
            phase = LauncherPhase.READY
        if phase == LauncherPhase.COOLDOWN and cooldown <= 0.0:
            phase = LauncherPhase.READY if flywheels_ready else LauncherPhase.SPINUP

        if request.trigger and phase == LauncherPhase.READY:
            if rejection_reasons:
                pending = None
            else:
                pending = request
                feeder_target += config.feeder_step_rad
                phase = LauncherPhase.FEEDING

        feeder_position, feeder_velocity, feeder_torque = self._integrate_feeder(
            state.feeder_position,
            state.feeder_velocity,
            feeder_target,
            dt,
        )

        launch = None
        if (
            phase == LauncherPhase.FEEDING
            and pending is not None
            and abs(feeder_target - feeder_position)
            <= config.feeder_position_tolerance_rad
            and abs(feeder_velocity) <= 1.0
        ):
            actual_speed = min(
                pending.requested_launch_speed_mps,
                0.5
                * (abs(left_velocity) + abs(right_velocity))
                * config.flywheel_radius_m
                * config.wheel_to_projectile_efficiency,
            )
            kinetic_energy = 0.5 * config.projectile_mass_kg * actual_speed**2
            launch = ProjectileLaunch(
                speed_mps=actual_speed,
                kinetic_energy_j=kinetic_energy,
                recoil_impulse_body_ns=np.asarray(
                    [-config.projectile_mass_kg * actual_speed, 0.0, 0.0],
                    dtype=np.float64,
                ),
                barrel_heat_after=cooled_heat + config.barrel_heat_per_projectile,
                ammunition_after=state.ammunition - 1,
            )
            energy_per_wheel = kinetic_energy / (
                2.0 * config.wheel_to_projectile_efficiency
            )
            left_velocity = self._remove_rotational_energy(
                left_velocity, energy_per_wheel
            )
            right_velocity = self._remove_rotational_energy(
                right_velocity, energy_per_wheel
            )
            cooled_heat = launch.barrel_heat_after
            cooldown = config.cooldown_s
            phase = LauncherPhase.COOLDOWN
            pending = None

        self._pending_request = pending
        self.state = replace(
            state,
            phase=phase,
            left_flywheel_velocity=left_velocity,
            right_flywheel_velocity=right_velocity,
            feeder_position=feeder_position,
            feeder_velocity=feeder_velocity,
            feeder_target=feeder_target,
            barrel_heat=cooled_heat,
            ammunition=(launch.ammunition_after if launch else state.ammunition),
            cooldown_remaining_s=cooldown,
            fault_reason="",
        )
        return LauncherStepResult(
            state=self.state,
            left_flywheel_torque_nm=float(left_torque),
            right_flywheel_torque_nm=float(right_torque),
            feeder_torque_nm=float(feeder_torque),
            launch=launch,
            rejection_reasons=rejection_reasons if request.trigger else (),
        )

    def _flywheel_velocity_torque(self, target: float, actual: float) -> float:
        return float(
            np.clip(
                0.012 * (target - actual),
                -self.config.maximum_flywheel_torque_nm,
                self.config.maximum_flywheel_torque_nm,
            )
        )

    def _integrate_flywheel(self, velocity: float, torque: float, dt: float) -> float:
        acceleration = (
            torque - self.config.flywheel_viscous_friction * velocity
        ) / self.config.flywheel_inertia_kg_m2
        # Semi-implicit Euler remains stable for the 2 ms shared physics step.
        return float(velocity + acceleration * dt)

    def _integrate_feeder(
        self,
        position: float,
        velocity: float,
        target: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """Integrate the stiff feeder servo without a 2 ms limit cycle.

        The feeder's reflected inertia is much smaller than a leg link's.  A
        single 2 ms semi-implicit step sits close to the stability boundary
        of its fastest damped pole, so the mechanism is integrated at no more
        than 1 ms while the robot physics contract remains at 2 ms.  Both
        backends use this shared mechanism state, keeping the substep choice
        out of the simulator-specific assets.
        """

        config = self.config
        substeps = max(1, math.ceil(dt / 0.001))
        substep_dt = dt / substeps
        torque_integral = 0.0
        for _ in range(substeps):
            torque = float(
                np.clip(
                    config.feeder_stiffness * (target - position)
                    - config.feeder_damping * velocity,
                    -config.maximum_feeder_torque_nm,
                    config.maximum_feeder_torque_nm,
                )
            )
            acceleration = (
                torque - config.feeder_viscous_friction * velocity
            ) / config.feeder_inertia_kg_m2
            velocity += acceleration * substep_dt
            position += velocity * substep_dt
            torque_integral += torque * substep_dt
        return float(position), float(velocity), float(torque_integral / dt)

    def _remove_rotational_energy(self, velocity: float, energy: float) -> float:
        remaining = max(
            0.0,
            0.5 * self.config.flywheel_inertia_kg_m2 * velocity**2 - energy,
        )
        return math.copysign(
            math.sqrt(2.0 * remaining / self.config.flywheel_inertia_kg_m2),
            velocity,
        )


@dataclass(frozen=True)
class BallisticConfig:
    projectile_mass_kg: float = RMUC_2026.projectile_mass_kg
    projectile_radius_m: float = 0.5 * RMUC_2026.projectile_diameter_m
    air_density_kg_m3: float = 1.225
    drag_coefficient: float = 0.47
    gravity_mps2: float = 9.81
    wind_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def frontal_area_m2(self) -> float:
        return math.pi * self.projectile_radius_m**2


def ballistic_acceleration(
    velocity_mps: np.ndarray,
    config: BallisticConfig = BallisticConfig(),
) -> np.ndarray:
    velocity = np.asarray(velocity_mps, dtype=np.float64)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError("velocity_mps must be a finite 3-vector")
    relative = velocity - np.asarray(config.wind_velocity_mps, dtype=np.float64)
    speed = np.linalg.norm(relative)
    drag_scale = (
        0.5
        * config.air_density_kg_m3
        * config.drag_coefficient
        * config.frontal_area_m2
        / config.projectile_mass_kg
    )
    return np.asarray(
        [0.0, 0.0, -config.gravity_mps2], dtype=np.float64
    ) - drag_scale * speed * relative


def integrate_ballistic_rk4(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    dt: float,
    config: BallisticConfig = BallisticConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Advance a spherical projectile with gravity and quadratic drag."""

    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(velocity_mps, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("position and velocity must be 3-vectors")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("ballistic state contains a non-finite value")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    def derivative(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del pos
        return vel, ballistic_acceleration(vel, config)

    p1, v1 = derivative(position, velocity)
    p2, v2 = derivative(position + 0.5 * dt * p1, velocity + 0.5 * dt * v1)
    p3, v3 = derivative(position + 0.5 * dt * p2, velocity + 0.5 * dt * v2)
    p4, v4 = derivative(position + dt * p3, velocity + dt * v3)
    next_position = position + dt * (p1 + 2 * p2 + 2 * p3 + p4) / 6.0
    next_velocity = velocity + dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6.0
    return next_position, next_velocity
