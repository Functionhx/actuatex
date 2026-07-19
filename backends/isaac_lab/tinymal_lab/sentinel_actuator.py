"""Isaac Lab explicit actuator backed by the shared Sentinel motor physics."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import DelayBuffer
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions

from tasks.robomaster.contract import LOCOMOTION_JOINT_NAMES, RMUC_2026, SIM_DT
from tasks.robomaster.torch_powertrain import (
    TorchRefereePowerMonitor,
    allocate_motor_power_torch,
    integrate_motor_temperature_torch,
    motor_parameters,
)


class SentinelDCMotor(IdealPDActuator):
    """Voltage/current/thermal-limited actuator with RMUC chassis power state."""

    cfg: SentinelDCMotorCfg

    def __init__(self, cfg: SentinelDCMotorCfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        unknown = sorted(set(self.joint_names) - set(cfg.expected_joint_names))
        if unknown:
            raise ValueError(f"Sentinel actuator received unknown joints: {unknown}")
        if not 0.0 < cfg.chassis_power_limit_w <= cfg.managed_power_ceiling_w:
            raise ValueError(
                "managed_power_ceiling_w must be no lower than the chassis limit"
            )
        self._parameters = motor_parameters(
            self.joint_names,
            device=self._device,
            dtype=self.computed_effort.dtype,
        )
        self._chassis_mask = torch.tensor(
            [name in LOCOMOTION_JOINT_NAMES for name in self.joint_names],
            device=self._device,
            dtype=torch.bool,
        )
        self.motor_temperature_c = torch.full_like(
            self.computed_effort, cfg.ambient_temperature_c
        )
        self.motor_current_a = torch.zeros_like(self.computed_effort)
        self.terminal_voltage_v = torch.zeros_like(self.computed_effort)
        self.electrical_power_w = torch.zeros_like(self.computed_effort)
        self.copper_loss_w = torch.zeros_like(self.computed_effort)
        self.torque_scale = torch.ones(self._num_envs, device=self._device)
        self.accounted_chassis_power_w = torch.zeros_like(self.torque_scale)
        self.referee = TorchRefereePowerMonitor(
            self._num_envs,
            device=self._device,
            dtype=self.computed_effort.dtype,
            power_limit_w=cfg.chassis_power_limit_w,
        )
        if not 0 <= cfg.minimum_command_delay_steps <= cfg.maximum_command_delay_steps:
            raise ValueError("invalid Sentinel command-delay range")
        self._all_env_ids = torch.arange(
            self._num_envs,
            device=self._device,
            dtype=torch.long,
        )
        self._position_delay = DelayBuffer(
            cfg.maximum_command_delay_steps,
            self._num_envs,
            device=self._device,
        )
        self._velocity_delay = DelayBuffer(
            cfg.maximum_command_delay_steps,
            self._num_envs,
            device=self._device,
        )
        self._effort_delay = DelayBuffer(
            cfg.maximum_command_delay_steps,
            self._num_envs,
            device=self._device,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> None:
        indices = slice(None) if env_ids is None else env_ids
        self.motor_temperature_c[indices] = self.cfg.ambient_temperature_c
        self.motor_current_a[indices] = 0.0
        self.terminal_voltage_v[indices] = 0.0
        self.electrical_power_w[indices] = 0.0
        self.copper_loss_w[indices] = 0.0
        self.torque_scale[indices] = 1.0
        self.accounted_chassis_power_w[indices] = 0.0
        self.referee.reset(indices)
        if env_ids is None or isinstance(env_ids, slice):
            count = self._num_envs
            delay_indices = self._all_env_ids
        else:
            count = len(env_ids)
            delay_indices = env_ids
        delay = torch.randint(
            low=self.cfg.minimum_command_delay_steps,
            high=self.cfg.maximum_command_delay_steps + 1,
            size=(count,),
            device=self._device,
            dtype=torch.int,
        )
        for buffer in (
            self._position_delay,
            self._velocity_delay,
            self._effort_delay,
        ):
            buffer.set_time_lag(delay, delay_indices)
            buffer.reset(delay_indices)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action.joint_positions = self._position_delay.compute(
            control_action.joint_positions
        )
        control_action.joint_velocities = self._velocity_delay.compute(
            control_action.joint_velocities
        )
        control_action.joint_efforts = self._effort_delay.compute(
            control_action.joint_efforts
        )
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        requested = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + control_action.joint_efforts
        )
        self.computed_effort = requested

        enabled = self.referee.chassis_enabled.unsqueeze(-1)
        requested = torch.where(
            self._chassis_mask & ~enabled,
            torch.zeros_like(requested),
            requested,
        )
        buffer_fraction = self.referee.buffer_energy_j / self.referee.buffer_limit_j
        managed_limit = self.cfg.chassis_power_limit_w + (
            self.cfg.managed_power_ceiling_w - self.cfg.chassis_power_limit_w
        ) * buffer_fraction
        managed_limit = torch.where(
            self.referee.chassis_enabled,
            managed_limit,
            torch.zeros_like(managed_limit),
        )
        allocation = allocate_motor_power_torch(
            requested,
            joint_vel,
            power_limit_w=managed_limit,
            power_mask=self._chassis_mask,
            joint_names=self.joint_names,
            bus_voltage_v=self.cfg.bus_voltage_v,
            temperature_c=self.motor_temperature_c,
            iterations=self.cfg.power_allocation_iterations,
            parameters=self._parameters,
        )
        self.applied_effort = allocation.motor.joint_torque
        self.motor_current_a = allocation.motor.motor_current_a
        self.terminal_voltage_v = allocation.motor.terminal_voltage_v
        self.electrical_power_w = allocation.motor.electrical_power_w
        self.copper_loss_w = allocation.motor.copper_loss_w
        self.torque_scale = allocation.torque_scale
        self.accounted_chassis_power_w = allocation.accounted_bus_power_w
        self.motor_temperature_c = integrate_motor_temperature_torch(
            self.motor_temperature_c,
            self.copper_loss_w,
            self.cfg.simulation_dt,
            joint_names=self.joint_names,
            ambient_temperature_c=self.cfg.ambient_temperature_c,
            parameters=self._parameters,
        )
        self.referee.step(
            self.accounted_chassis_power_w,
            self.cfg.simulation_dt,
        )

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class SentinelDCMotorCfg(IdealPDActuatorCfg):
    """Configuration for the full shared Sentinel actuator bank."""

    class_type: type[SentinelDCMotor] = SentinelDCMotor
    expected_joint_names: tuple[str, ...] = ()
    simulation_dt: float = SIM_DT
    bus_voltage_v: float = 24.0
    ambient_temperature_c: float = 25.0
    chassis_power_limit_w: float = RMUC_2026.full_auto_chassis_power_limit_w
    managed_power_ceiling_w: float = 180.0
    power_allocation_iterations: int = 18
    minimum_command_delay_steps: int = 0
    maximum_command_delay_steps: int = 0
