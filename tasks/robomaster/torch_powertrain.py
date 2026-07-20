"""Torch twin of the Sentinel electrical, thermal and power-limit model.

The equations intentionally mirror :mod:`tasks.robomaster.powertrain`. Isaac
Lab uses this implementation directly on the GPU; MuJoCo and deployment tests
use the NumPy reference. Keeping both behind parity tests prevents a silent
change in motor semantics when moving a policy between simulators.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .contract import ALL_JOINT_NAMES, MOTOR_SPECS, RMUC_2026


@dataclass(frozen=True)
class TorchMotorParameters:
    gear_ratio: torch.Tensor
    torque_constant: torch.Tensor
    back_emf: torch.Tensor
    resistance: torch.Tensor
    current_limit: torch.Tensor
    torque_limit: torch.Tensor
    transmission_efficiency: torch.Tensor
    viscous_friction: torch.Tensor
    inverter_efficiency: torch.Tensor
    regenerative_efficiency: torch.Tensor
    derate_temperature: torch.Tensor
    shutdown_temperature: torch.Tensor
    thermal_resistance: torch.Tensor
    thermal_capacitance: torch.Tensor


@dataclass(frozen=True)
class TorchMotorBankResult:
    joint_torque: torch.Tensor
    motor_current_a: torch.Tensor
    terminal_voltage_v: torch.Tensor
    electrical_power_w: torch.Tensor
    mechanical_power_w: torch.Tensor
    copper_loss_w: torch.Tensor
    current_limit_a: torch.Tensor

    @property
    def positive_bus_power_w(self) -> torch.Tensor:
        return self.electrical_power_w.clamp_min(0.0).sum(dim=-1)


@dataclass(frozen=True)
class TorchPowerAllocationResult:
    motor: TorchMotorBankResult
    torque_scale: torch.Tensor
    power_limit_w: torch.Tensor
    accounted_bus_power_w: torch.Tensor


def motor_parameters(
    joint_names: Sequence[str] = ALL_JOINT_NAMES,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> TorchMotorParameters:
    names = tuple(joint_names)
    if len(set(names)) != len(names):
        raise ValueError("joint_names contains duplicates")
    unknown = sorted(set(names) - set(ALL_JOINT_NAMES))
    if unknown:
        raise ValueError(f"unknown Sentinel joints: {unknown}")
    by_name = dict(zip(ALL_JOINT_NAMES, MOTOR_SPECS))
    specs = [by_name[name] for name in names]

    def tensor(attribute: str) -> torch.Tensor:
        return torch.tensor(
            [getattr(spec, attribute) for spec in specs],
            device=device,
            dtype=dtype,
        )

    return TorchMotorParameters(
        gear_ratio=tensor("gear_ratio"),
        torque_constant=tensor("torque_constant_nm_per_a"),
        back_emf=tensor("back_emf_v_per_rad_s"),
        resistance=tensor("resistance_ohm"),
        current_limit=tensor("current_limit_a"),
        torque_limit=tensor("joint_torque_limit_nm"),
        transmission_efficiency=tensor("transmission_efficiency"),
        viscous_friction=tensor("viscous_friction_nm_per_rad_s"),
        inverter_efficiency=tensor("inverter_efficiency"),
        regenerative_efficiency=tensor("regenerative_efficiency"),
        derate_temperature=tensor("derate_temperature_c"),
        shutdown_temperature=tensor("shutdown_temperature_c"),
        thermal_resistance=tensor("thermal_resistance_k_per_w"),
        thermal_capacitance=tensor("thermal_capacitance_j_per_k"),
    )


def _validate_motor_tensor(value: torch.Tensor, joint_count: int, label: str) -> None:
    if value.ndim == 0 or value.shape[-1] != joint_count:
        raise ValueError(
            f"{label} must have last dimension {joint_count}, got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains a non-finite value")


def step_motor_bank_torch(
    requested_joint_torque: torch.Tensor,
    joint_velocity: torch.Tensor,
    *,
    joint_names: Sequence[str] = ALL_JOINT_NAMES,
    bus_voltage_v: float | torch.Tensor = 24.0,
    temperature_c: torch.Tensor | None = None,
    parameters: TorchMotorParameters | None = None,
) -> TorchMotorBankResult:
    """Apply the same electrical and thermal limits as the NumPy reference."""

    names = tuple(joint_names)
    _validate_motor_tensor(requested_joint_torque, len(names), "requested_joint_torque")
    _validate_motor_tensor(joint_velocity, len(names), "joint_velocity")
    if requested_joint_torque.shape != joint_velocity.shape:
        raise ValueError("requested torque and joint velocity shapes must match")
    if requested_joint_torque.device != joint_velocity.device:
        raise ValueError("requested torque and joint velocity devices must match")
    if parameters is None:
        parameters = motor_parameters(
            names,
            device=requested_joint_torque.device,
            dtype=requested_joint_torque.dtype,
        )
    if temperature_c is None:
        temperature = torch.full_like(requested_joint_torque, 25.0)
    else:
        _validate_motor_tensor(temperature_c, len(names), "temperature_c")
        if temperature_c.shape != requested_joint_torque.shape:
            raise ValueError("temperature and requested torque shapes must match")
        temperature = temperature_c

    leading_shape = requested_joint_torque.shape[:-1]
    bus_voltage = torch.as_tensor(
        bus_voltage_v,
        device=requested_joint_torque.device,
        dtype=requested_joint_torque.dtype,
    ).broadcast_to(leading_shape)
    if not bool(torch.isfinite(bus_voltage).all()) or bool((bus_voltage <= 0.0).any()):
        raise ValueError("bus_voltage_v must be finite and positive")

    requested = torch.clamp(
        requested_joint_torque,
        min=-parameters.torque_limit,
        max=parameters.torque_limit,
    )
    motor_velocity = joint_velocity * parameters.gear_ratio
    thermal_scale = torch.clamp(
        (parameters.shutdown_temperature - temperature)
        / (parameters.shutdown_temperature - parameters.derate_temperature),
        min=0.0,
        max=1.0,
    )
    effective_current_limit = parameters.current_limit * thermal_scale
    desired_current = requested / (
        parameters.torque_constant
        * parameters.gear_ratio
        * parameters.transmission_efficiency
    )
    back_emf_voltage = parameters.back_emf * motor_velocity
    positive_voltage_current = torch.clamp(
        (bus_voltage.unsqueeze(-1) - back_emf_voltage) / parameters.resistance,
        min=0.0,
    )
    negative_voltage_current = torch.clamp(
        (-bus_voltage.unsqueeze(-1) - back_emf_voltage) / parameters.resistance,
        max=0.0,
    )
    lower = torch.maximum(-effective_current_limit, negative_voltage_current)
    upper = torch.minimum(effective_current_limit, positive_voltage_current)
    current = torch.clamp(desired_current, min=lower, max=upper)
    joint_torque = (
        current
        * parameters.torque_constant
        * parameters.gear_ratio
        * parameters.transmission_efficiency
        - parameters.viscous_friction * joint_velocity
    )
    joint_torque = torch.clamp(
        joint_torque,
        min=-parameters.torque_limit,
        max=parameters.torque_limit,
    )
    terminal_voltage = back_emf_voltage + current * parameters.resistance
    terminal_power = terminal_voltage * current
    electrical_power = torch.where(
        terminal_power >= 0.0,
        terminal_power / parameters.inverter_efficiency,
        terminal_power * parameters.regenerative_efficiency,
    )
    return TorchMotorBankResult(
        joint_torque=joint_torque,
        motor_current_a=current,
        terminal_voltage_v=terminal_voltage,
        electrical_power_w=electrical_power,
        mechanical_power_w=joint_torque * joint_velocity,
        copper_loss_w=current.square() * parameters.resistance,
        current_limit_a=effective_current_limit,
    )


def allocate_motor_power_torch(
    requested_joint_torque: torch.Tensor,
    joint_velocity: torch.Tensor,
    *,
    power_limit_w: float | torch.Tensor,
    power_mask: torch.Tensor | Sequence[bool] | None = None,
    joint_names: Sequence[str] = ALL_JOINT_NAMES,
    bus_voltage_v: float | torch.Tensor = 24.0,
    temperature_c: torch.Tensor | None = None,
    iterations: int = 24,
    parameters: TorchMotorParameters | None = None,
) -> TorchPowerAllocationResult:
    """Scale the selected subsystem until its positive bus draw meets a limit."""

    names = tuple(joint_names)
    _validate_motor_tensor(requested_joint_torque, len(names), "requested_joint_torque")
    _validate_motor_tensor(joint_velocity, len(names), "joint_velocity")
    leading_shape = requested_joint_torque.shape[:-1]
    limit = torch.as_tensor(
        power_limit_w,
        device=requested_joint_torque.device,
        dtype=requested_joint_torque.dtype,
    ).broadcast_to(leading_shape)
    if not bool(torch.isfinite(limit).all()) or bool((limit < 0.0).any()):
        raise ValueError("power_limit_w must be finite and non-negative")
    if power_mask is None:
        mask = torch.ones(
            len(names), device=requested_joint_torque.device, dtype=torch.bool
        )
    else:
        mask = torch.as_tensor(
            power_mask,
            device=requested_joint_torque.device,
            dtype=torch.bool,
        )
        if mask.shape != (len(names),):
            raise ValueError(f"power_mask must have shape {(len(names),)}")
    if parameters is None:
        parameters = motor_parameters(
            names,
            device=requested_joint_torque.device,
            dtype=requested_joint_torque.dtype,
        )

    def evaluate(scale: torch.Tensor) -> tuple[TorchMotorBankResult, torch.Tensor]:
        scaled = torch.where(
            mask,
            requested_joint_torque * scale.unsqueeze(-1),
            requested_joint_torque,
        )
        motor = step_motor_bank_torch(
            scaled,
            joint_velocity,
            joint_names=names,
            bus_voltage_v=bus_voltage_v,
            temperature_c=temperature_c,
            parameters=parameters,
        )
        accounted = torch.where(
            mask,
            motor.electrical_power_w.clamp_min(0.0),
            torch.zeros_like(motor.electrical_power_w),
        ).sum(dim=-1)
        return motor, accounted

    ones = torch.ones(leading_shape, device=limit.device, dtype=limit.dtype)
    unconstrained, unconstrained_power = evaluate(ones)
    needs_scaling = unconstrained_power > limit
    low = torch.zeros_like(limit)
    high = torch.ones_like(limit)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        _, candidate_power = evaluate(middle)
        feasible = candidate_power <= limit
        low = torch.where(feasible, middle, low)
        high = torch.where(feasible, high, middle)
    scale = torch.where(needs_scaling, low, ones)
    allocated, accounted = evaluate(scale)
    return TorchPowerAllocationResult(
        motor=allocated,
        torque_scale=scale,
        power_limit_w=limit,
        accounted_bus_power_w=accounted,
    )


def integrate_motor_temperature_torch(
    temperature_c: torch.Tensor,
    copper_loss_w: torch.Tensor,
    dt: float,
    *,
    joint_names: Sequence[str] = ALL_JOINT_NAMES,
    ambient_temperature_c: float = 25.0,
    parameters: TorchMotorParameters | None = None,
) -> torch.Tensor:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    names = tuple(joint_names)
    _validate_motor_tensor(temperature_c, len(names), "temperature_c")
    _validate_motor_tensor(copper_loss_w, len(names), "copper_loss_w")
    if temperature_c.shape != copper_loss_w.shape:
        raise ValueError("temperature and copper loss shapes must match")
    if parameters is None:
        parameters = motor_parameters(
            names,
            device=temperature_c.device,
            dtype=temperature_c.dtype,
        )
    cooling = (
        temperature_c - ambient_temperature_c
    ) / parameters.thermal_resistance
    derivative = (
        copper_loss_w - cooling
    ) / parameters.thermal_capacitance
    return torch.clamp_min(temperature_c + dt * derivative, ambient_temperature_c)


class TorchRefereePowerMonitor:
    """Batched 10 Hz referee-system state for GPU-parallel environments."""

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        power_limit_w: float = RMUC_2026.full_auto_chassis_power_limit_w,
        buffer_limit_j: float = RMUC_2026.chassis_buffer_energy_limit_j,
        sample_period_s: float = RMUC_2026.referee_detection_period_s,
        power_off_duration_s: float = RMUC_2026.chassis_power_off_duration_s,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if min(power_limit_w, buffer_limit_j, sample_period_s, power_off_duration_s) <= 0.0:
            raise ValueError("referee power parameters must be positive")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.power_limit_w = float(power_limit_w)
        self.buffer_limit_j = float(buffer_limit_j)
        self.sample_period_s = float(sample_period_s)
        self.power_off_duration_s = float(power_off_duration_s)
        self.buffer_energy_j = torch.full(
            (num_envs,), buffer_limit_j, device=self.device, dtype=dtype
        )
        self.powered_off_remaining_s = torch.zeros(
            num_envs, device=self.device, dtype=dtype
        )
        self.sample_elapsed_s = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self.sample_energy_j = torch.zeros_like(self.sample_elapsed_s)

    @property
    def chassis_enabled(self) -> torch.Tensor:
        return self.powered_off_remaining_s <= 0.0

    def reset(self, env_ids: torch.Tensor | Sequence[int] | slice | None = None) -> None:
        indices = slice(None) if env_ids is None else env_ids
        self.buffer_energy_j[indices] = self.buffer_limit_j
        self.powered_off_remaining_s[indices] = 0.0
        self.sample_elapsed_s[indices] = 0.0
        self.sample_energy_j[indices] = 0.0

    def step(self, measured_chassis_power_w: torch.Tensor, dt: float) -> None:
        if measured_chassis_power_w.shape != (self.num_envs,):
            raise ValueError(
                "measured_chassis_power_w must have shape "
                f"{(self.num_envs,)}, got {tuple(measured_chassis_power_w.shape)}"
            )
        if not bool(torch.isfinite(measured_chassis_power_w).all()) or bool(
            (measured_chassis_power_w < 0.0).any()
        ):
            raise ValueError("measured_chassis_power_w must be finite and non-negative")
        if not 0.0 < dt <= self.sample_period_s:
            raise ValueError("dt must be in (0, sample_period_s]")

        enabled_at_start = self.chassis_enabled
        effective_power = torch.where(
            enabled_at_start,
            measured_chassis_power_w,
            torch.zeros_like(measured_chassis_power_w),
        )
        self.sample_energy_j += effective_power * dt
        self.sample_elapsed_s += dt
        self.powered_off_remaining_s = torch.clamp_min(
            self.powered_off_remaining_s - dt, 0.0
        )
        self.powered_off_remaining_s = torch.where(
            self.powered_off_remaining_s <= 1.0e-6,
            torch.zeros_like(self.powered_off_remaining_s),
            self.powered_off_remaining_s,
        )

        sampled = self.sample_elapsed_s >= self.sample_period_s - 1.0e-6
        average_power = self.sample_energy_j / self.sample_period_s
        next_buffer = torch.clamp(
            self.buffer_energy_j
            - (average_power - self.power_limit_w) * self.sample_period_s,
            min=0.0,
            max=self.buffer_limit_j,
        )
        self.buffer_energy_j = torch.where(sampled, next_buffer, self.buffer_energy_j)
        depleted = self.buffer_energy_j <= 1.0e-6
        self.buffer_energy_j = torch.where(
            depleted, torch.zeros_like(self.buffer_energy_j), self.buffer_energy_j
        )
        cutoff = sampled & depleted & (average_power > self.power_limit_w) & self.chassis_enabled
        self.powered_off_remaining_s = torch.where(
            cutoff,
            torch.full_like(
                self.powered_off_remaining_s, self.power_off_duration_s
            ),
            self.powered_off_remaining_s,
        )
        self.sample_elapsed_s = torch.where(
            sampled,
            torch.zeros_like(self.sample_elapsed_s),
            self.sample_elapsed_s,
        )
        self.sample_energy_j = torch.where(
            sampled,
            torch.zeros_like(self.sample_energy_j),
            self.sample_energy_j,
        )
