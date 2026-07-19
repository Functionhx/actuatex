"""Electrical, thermal and referee-power dynamics for Sentinel actuators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import DCMotorSpec, MOTOR_SPECS, RMUC_2026


def _motor_array(values: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0 or result.shape[-1] != len(MOTOR_SPECS):
        raise ValueError(
            f"{label} must have last dimension {len(MOTOR_SPECS)}, "
            f"got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result


@dataclass(frozen=True)
class MotorBankResult:
    joint_torque: np.ndarray
    motor_current_a: np.ndarray
    terminal_voltage_v: np.ndarray
    electrical_power_w: np.ndarray
    mechanical_power_w: np.ndarray
    copper_loss_w: np.ndarray
    current_limit_a: np.ndarray

    @property
    def positive_bus_power_w(self) -> np.ndarray:
        return np.maximum(self.electrical_power_w, 0.0).sum(axis=-1)


@dataclass(frozen=True)
class PowerAllocationResult:
    motor: MotorBankResult
    torque_scale: np.ndarray
    power_limit_w: np.ndarray
    accounted_bus_power_w: np.ndarray


def temperature_current_scale(
    temperature_c: np.ndarray,
    spec: DCMotorSpec,
) -> np.ndarray:
    """Linear current derating from the warning to shutdown temperature."""

    temperature_c = np.asarray(temperature_c, dtype=np.float64)
    scale = (spec.shutdown_temperature_c - temperature_c) / (
        spec.shutdown_temperature_c - spec.derate_temperature_c
    )
    return np.clip(scale, 0.0, 1.0)


def step_motor_bank(
    requested_joint_torque: np.ndarray,
    joint_velocity: np.ndarray,
    *,
    bus_voltage_v: float | np.ndarray = 24.0,
    temperature_c: np.ndarray | None = None,
) -> MotorBankResult:
    """Apply current, voltage, speed, thermal and mechanical motor limits.

    A zero torque request is allowed to coast even when back EMF exceeds the
    supply.  A braking request can regenerate, while positive bus power includes
    inverter loss and negative power is reduced by regenerative efficiency.
    """

    requested = _motor_array(requested_joint_torque, "requested_joint_torque")
    velocity = _motor_array(joint_velocity, "joint_velocity")
    if requested.shape != velocity.shape:
        raise ValueError("requested torque and joint velocity shapes must match")
    if temperature_c is None:
        temperature = np.full_like(requested, 25.0)
    else:
        temperature = _motor_array(temperature_c, "temperature_c")
        if temperature.shape != requested.shape:
            raise ValueError("temperature and requested torque shapes must match")

    leading_shape = requested.shape[:-1]
    bus_voltage = np.broadcast_to(
        np.asarray(bus_voltage_v, dtype=np.float64), leading_shape
    )
    if not np.isfinite(bus_voltage).all() or np.any(bus_voltage <= 0.0):
        raise ValueError("bus_voltage_v must be finite and positive")

    torque = np.empty_like(requested)
    current = np.empty_like(requested)
    terminal_voltage = np.empty_like(requested)
    electrical_power = np.empty_like(requested)
    mechanical_power = np.empty_like(requested)
    copper_loss = np.empty_like(requested)
    current_limit = np.empty_like(requested)

    for index, spec in enumerate(MOTOR_SPECS):
        requested_i = np.clip(
            requested[..., index],
            -spec.joint_torque_limit_nm,
            spec.joint_torque_limit_nm,
        )
        joint_velocity_i = velocity[..., index]
        motor_velocity = joint_velocity_i * spec.gear_ratio
        effective_current_limit = spec.current_limit_a * temperature_current_scale(
            temperature[..., index], spec
        )
        desired_current = requested_i / (
            spec.torque_constant_nm_per_a
            * spec.gear_ratio
            * spec.transmission_efficiency
        )

        positive_voltage_current = np.maximum(
            (bus_voltage - spec.back_emf_v_per_rad_s * motor_velocity)
            / spec.resistance_ohm,
            0.0,
        )
        negative_voltage_current = np.minimum(
            (-bus_voltage - spec.back_emf_v_per_rad_s * motor_velocity)
            / spec.resistance_ohm,
            0.0,
        )
        lower = np.maximum(-effective_current_limit, negative_voltage_current)
        upper = np.minimum(effective_current_limit, positive_voltage_current)
        motor_current = np.clip(desired_current, lower, upper)
        joint_torque = (
            motor_current
            * spec.torque_constant_nm_per_a
            * spec.gear_ratio
            * spec.transmission_efficiency
        )
        joint_torque -= spec.viscous_friction_nm_per_rad_s * joint_velocity_i
        joint_torque = np.clip(
            joint_torque,
            -spec.joint_torque_limit_nm,
            spec.joint_torque_limit_nm,
        )

        voltage = (
            spec.back_emf_v_per_rad_s * motor_velocity
            + motor_current * spec.resistance_ohm
        )
        terminal_power = voltage * motor_current
        bus_power = np.where(
            terminal_power >= 0.0,
            terminal_power / spec.inverter_efficiency,
            terminal_power * spec.regenerative_efficiency,
        )

        torque[..., index] = joint_torque
        current[..., index] = motor_current
        terminal_voltage[..., index] = voltage
        electrical_power[..., index] = bus_power
        mechanical_power[..., index] = joint_torque * joint_velocity_i
        copper_loss[..., index] = np.square(motor_current) * spec.resistance_ohm
        current_limit[..., index] = effective_current_limit

    return MotorBankResult(
        joint_torque=torque,
        motor_current_a=current,
        terminal_voltage_v=terminal_voltage,
        electrical_power_w=electrical_power,
        mechanical_power_w=mechanical_power,
        copper_loss_w=copper_loss,
        current_limit_a=current_limit,
    )


def allocate_motor_power(
    requested_joint_torque: np.ndarray,
    joint_velocity: np.ndarray,
    *,
    power_limit_w: float | np.ndarray,
    power_mask: np.ndarray | None = None,
    bus_voltage_v: float | np.ndarray = 24.0,
    temperature_c: np.ndarray | None = None,
    iterations: int = 36,
) -> PowerAllocationResult:
    """Scale a selected subsystem until its positive bus draw meets a limit."""

    requested = _motor_array(requested_joint_torque, "requested_joint_torque")
    velocity = _motor_array(joint_velocity, "joint_velocity")
    leading_shape = requested.shape[:-1]
    limit = np.broadcast_to(np.asarray(power_limit_w, dtype=np.float64), leading_shape)
    if not np.isfinite(limit).all() or np.any(limit < 0.0):
        raise ValueError("power_limit_w must be finite and non-negative")
    if power_mask is None:
        mask = np.ones(len(MOTOR_SPECS), dtype=bool)
    else:
        mask = np.asarray(power_mask, dtype=bool)
        if mask.shape != (len(MOTOR_SPECS),):
            raise ValueError(
                f"power_mask must have shape {(len(MOTOR_SPECS),)}, got {mask.shape}"
            )

    def evaluate(scale: np.ndarray) -> tuple[MotorBankResult, np.ndarray]:
        scaled = np.where(
            mask,
            requested * scale[..., np.newaxis],
            requested,
        )
        motor = step_motor_bank(
            scaled,
            velocity,
            bus_voltage_v=bus_voltage_v,
            temperature_c=temperature_c,
        )
        accounted = np.where(
            mask,
            np.maximum(motor.electrical_power_w, 0.0),
            0.0,
        ).sum(axis=-1)
        return motor, accounted

    unconstrained, unconstrained_power = evaluate(
        np.ones(leading_shape, dtype=np.float64)
    )
    needs_scaling = unconstrained_power > limit
    low = np.zeros(leading_shape, dtype=np.float64)
    high = np.ones(leading_shape, dtype=np.float64)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        _, candidate_power = evaluate(middle)
        feasible = candidate_power <= limit
        low = np.where(feasible, middle, low)
        high = np.where(feasible, high, middle)
    scale = np.where(needs_scaling, low, 1.0)
    allocated, accounted_power = evaluate(scale)
    return PowerAllocationResult(
        motor=allocated,
        torque_scale=scale,
        power_limit_w=limit,
        accounted_bus_power_w=accounted_power,
    )


def integrate_motor_temperature(
    temperature_c: np.ndarray,
    copper_loss_w: np.ndarray,
    dt: float,
    *,
    ambient_temperature_c: float = 25.0,
) -> np.ndarray:
    """Integrate a per-motor first-order lumped thermal model."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    temperature = _motor_array(temperature_c, "temperature_c")
    copper_loss = _motor_array(copper_loss_w, "copper_loss_w")
    if temperature.shape != copper_loss.shape:
        raise ValueError("temperature and copper loss shapes must match")
    derivative = np.empty_like(temperature)
    for index, spec in enumerate(MOTOR_SPECS):
        cooling = (
            temperature[..., index] - ambient_temperature_c
        ) / spec.thermal_resistance_k_per_w
        derivative[..., index] = (
            copper_loss[..., index] - cooling
        ) / spec.thermal_capacitance_j_per_k
    return np.maximum(
        ambient_temperature_c,
        temperature + dt * derivative,
    )


@dataclass
class RefereePowerState:
    buffer_energy_j: float
    powered_off_remaining_s: float = 0.0
    sample_elapsed_s: float = 0.0
    sample_energy_j: float = 0.0


class RefereePowerMonitor:
    """Exact 10 Hz RMUC chassis buffer-energy and power-off state machine."""

    def __init__(
        self,
        power_limit_w: float = RMUC_2026.full_auto_chassis_power_limit_w,
        buffer_limit_j: float = RMUC_2026.chassis_buffer_energy_limit_j,
        sample_period_s: float = RMUC_2026.referee_detection_period_s,
        power_off_duration_s: float = RMUC_2026.chassis_power_off_duration_s,
    ) -> None:
        if min(power_limit_w, buffer_limit_j, sample_period_s, power_off_duration_s) <= 0.0:
            raise ValueError("referee power parameters must be positive")
        self.power_limit_w = float(power_limit_w)
        self.buffer_limit_j = float(buffer_limit_j)
        self.sample_period_s = float(sample_period_s)
        self.power_off_duration_s = float(power_off_duration_s)
        self.state = RefereePowerState(buffer_energy_j=self.buffer_limit_j)

    @property
    def chassis_enabled(self) -> bool:
        return self.state.powered_off_remaining_s <= 0.0

    def reset(self) -> None:
        self.state = RefereePowerState(buffer_energy_j=self.buffer_limit_j)

    def step(self, measured_chassis_power_w: float, dt: float) -> RefereePowerState:
        if not np.isfinite(measured_chassis_power_w) or measured_chassis_power_w < 0.0:
            raise ValueError("measured_chassis_power_w must be finite and non-negative")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        remaining = float(dt)
        while remaining > 1.0e-12:
            chunk = min(
                remaining,
                self.sample_period_s - self.state.sample_elapsed_s,
            )
            effective_power = measured_chassis_power_w if self.chassis_enabled else 0.0
            self.state.sample_energy_j += effective_power * chunk
            self.state.sample_elapsed_s += chunk
            self.state.powered_off_remaining_s = max(
                0.0, self.state.powered_off_remaining_s - chunk
            )
            if self.state.powered_off_remaining_s <= 1.0e-9:
                self.state.powered_off_remaining_s = 0.0
            remaining -= chunk

            if self.state.sample_elapsed_s >= self.sample_period_s - 1.0e-12:
                average_power = self.state.sample_energy_j / self.sample_period_s
                self.state.buffer_energy_j -= (
                    average_power - self.power_limit_w
                ) * self.sample_period_s
                self.state.buffer_energy_j = float(
                    np.clip(
                        self.state.buffer_energy_j,
                        0.0,
                        self.buffer_limit_j,
                    )
                )
                # Six exact 100 ms samples at 100 W excess must consume the
                # 60 J buffer.  Remove binary floating-point residue before
                # evaluating the referee-system cutoff condition.
                if self.state.buffer_energy_j <= 1.0e-9:
                    self.state.buffer_energy_j = 0.0
                if (
                    self.state.buffer_energy_j <= 0.0
                    and average_power > self.power_limit_w
                    and self.chassis_enabled
                ):
                    self.state.powered_off_remaining_s = self.power_off_duration_s
                self.state.sample_elapsed_s = 0.0
                self.state.sample_energy_j = 0.0
        return self.state
