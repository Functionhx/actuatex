"""Shared contract for the ActuateX Sentinel wheel-legged robot.

The competition constants are transcribed from the official RMUC 2026 rule
manual pinned in ``robots/robomaster/config/rmuc_2026_v1.4.2.json``.  Motor
constants are ActuateX simulation nominals, not claims about a particular DJI
motor; they must be replaced by dynamometer identification before hardware use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


LOCOMOTION_JOINT_NAMES = (
    "left_hip_joint",
    "left_knee_joint",
    "right_hip_joint",
    "right_knee_joint",
    "left_wheel_joint",
    "right_wheel_joint",
)
GIMBAL_JOINT_NAMES = ("gimbal_yaw_joint", "gimbal_pitch_joint")
LAUNCHER_JOINT_NAMES = (
    "left_flywheel_joint",
    "right_flywheel_joint",
    "feeder_joint",
)
ALL_JOINT_NAMES = (
    LOCOMOTION_JOINT_NAMES + GIMBAL_JOINT_NAMES + LAUNCHER_JOINT_NAMES
)
# URDF declarations follow the two serial kinematic chains. Backends must map
# by name instead of assuming this storage order equals the policy order.
URDF_DECLARATION_JOINT_NAMES = (
    "left_hip_joint",
    "left_knee_joint",
    "left_wheel_joint",
    "right_hip_joint",
    "right_knee_joint",
    "right_wheel_joint",
    "gimbal_yaw_joint",
    "gimbal_pitch_joint",
    "left_flywheel_joint",
    "right_flywheel_joint",
    "feeder_joint",
)
POLICY_FROM_URDF = np.asarray(
    [URDF_DECLARATION_JOINT_NAMES.index(name) for name in ALL_JOINT_NAMES],
    dtype=np.int64,
)
URDF_FROM_POLICY = np.argsort(POLICY_FROM_URDF)

ACTION_DIM = len(LOCOMOTION_JOINT_NAMES)
FULL_ACTUATOR_DIM = len(ALL_JOINT_NAMES)
SIM_DT = 0.002
POLICY_DECIMATION = 10
POLICY_DT = SIM_DT * POLICY_DECIMATION
POLICY_FREQUENCY_HZ = 1.0 / POLICY_DT

SENTINEL_DEFAULT_JOINT_POSITION = np.asarray(
    [0.35, -0.70, 0.35, -0.70, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Rmuc2026SentryRules:
    ruleset: str = "RMUC 2026 Rule Manual V1.4.2 (2026-04-30)"
    projectile_diameter_m: float = 0.0168
    projectile_diameter_tolerance_m: float = 0.0002
    projectile_mass_kg: float = 0.0032
    projectile_mass_tolerance_kg: float = 0.0001
    initial_launch_speed_limit_mps: float = 25.0
    barrel_heat_per_projectile: float = 10.0
    full_auto_barrel_heat_limit: float = 260.0
    full_auto_cooling_per_second: float = 30.0
    full_auto_chassis_power_limit_w: float = 100.0
    semi_auto_barrel_heat_limit: float = 100.0
    semi_auto_cooling_per_second: float = 10.0
    semi_auto_chassis_power_limit_w: float = 60.0
    referee_detection_period_s: float = 0.1
    chassis_buffer_energy_limit_j: float = 60.0
    chassis_power_off_duration_s: float = 5.0


RMUC_2026 = Rmuc2026SentryRules()


@dataclass(frozen=True)
class DCMotorSpec:
    """Joint-side contract for a geared permanent-magnet DC motor."""

    name: str
    gear_ratio: float
    torque_constant_nm_per_a: float
    back_emf_v_per_rad_s: float
    resistance_ohm: float
    current_limit_a: float
    joint_torque_limit_nm: float
    joint_speed_limit_rad_s: float
    transmission_efficiency: float
    rotor_inertia_kg_m2: float
    viscous_friction_nm_per_rad_s: float
    thermal_resistance_k_per_w: float
    thermal_capacitance_j_per_k: float
    derate_temperature_c: float = 80.0
    shutdown_temperature_c: float = 110.0
    inverter_efficiency: float = 0.95
    regenerative_efficiency: float = 0.70

    @property
    def reflected_armature(self) -> float:
        return self.rotor_inertia_kg_m2 * self.gear_ratio**2

    @property
    def nominal_stall_joint_torque(self) -> float:
        electrical = (
            self.torque_constant_nm_per_a
            * self.current_limit_a
            * self.gear_ratio
            * self.transmission_efficiency
        )
        return min(electrical, self.joint_torque_limit_nm)

    @property
    def nominal_no_load_joint_speed(self) -> float:
        return min(
            24.0 / (self.back_emf_v_per_rad_s * self.gear_ratio),
            self.joint_speed_limit_rad_s,
        )


# ActuateX nominal motor families.  Values are deliberately centralized so a
# dyno fit changes both simulators and the deployment reference together.
LEG_MOTOR = DCMotorSpec(
    name="AX-Leg-10",
    gear_ratio=10.0,
    torque_constant_nm_per_a=0.105,
    back_emf_v_per_rad_s=0.105,
    resistance_ohm=0.16,
    current_limit_a=32.0,
    joint_torque_limit_nm=30.0,
    joint_speed_limit_rad_s=20.0,
    transmission_efficiency=0.88,
    rotor_inertia_kg_m2=5.0e-5,
    viscous_friction_nm_per_rad_s=0.015,
    thermal_resistance_k_per_w=1.6,
    thermal_capacitance_j_per_k=75.0,
)
WHEEL_MOTOR = DCMotorSpec(
    name="AX-Wheel-19",
    gear_ratio=19.2,
    torque_constant_nm_per_a=0.022,
    back_emf_v_per_rad_s=0.022,
    resistance_ohm=0.07,
    current_limit_a=32.0,
    joint_torque_limit_nm=12.0,
    joint_speed_limit_rad_s=50.0,
    transmission_efficiency=0.90,
    rotor_inertia_kg_m2=3.0e-5,
    viscous_friction_nm_per_rad_s=0.006,
    thermal_resistance_k_per_w=1.2,
    thermal_capacitance_j_per_k=95.0,
)
GIMBAL_MOTOR = DCMotorSpec(
    name="AX-Gimbal-6",
    gear_ratio=6.0,
    torque_constant_nm_per_a=0.040,
    back_emf_v_per_rad_s=0.040,
    resistance_ohm=0.20,
    current_limit_a=10.0,
    joint_torque_limit_nm=2.0,
    joint_speed_limit_rad_s=12.0,
    transmission_efficiency=0.86,
    rotor_inertia_kg_m2=2.0e-5,
    viscous_friction_nm_per_rad_s=0.008,
    thermal_resistance_k_per_w=2.2,
    thermal_capacitance_j_per_k=45.0,
)
FLYWHEEL_MOTOR = DCMotorSpec(
    name="AX-Flywheel-1",
    gear_ratio=1.0,
    torque_constant_nm_per_a=0.020,
    back_emf_v_per_rad_s=0.020,
    resistance_ohm=0.09,
    current_limit_a=24.0,
    joint_torque_limit_nm=0.45,
    joint_speed_limit_rad_s=1200.0,
    transmission_efficiency=0.95,
    rotor_inertia_kg_m2=2.0e-5,
    viscous_friction_nm_per_rad_s=2.0e-4,
    thermal_resistance_k_per_w=1.8,
    thermal_capacitance_j_per_k=35.0,
)
FEEDER_MOTOR = DCMotorSpec(
    name="AX-Feeder-10",
    gear_ratio=10.0,
    torque_constant_nm_per_a=0.065,
    back_emf_v_per_rad_s=0.065,
    resistance_ohm=0.15,
    current_limit_a=10.0,
    joint_torque_limit_nm=5.0,
    joint_speed_limit_rad_s=8.0,
    transmission_efficiency=0.82,
    rotor_inertia_kg_m2=8.3e-7,
    viscous_friction_nm_per_rad_s=0.020,
    thermal_resistance_k_per_w=2.0,
    thermal_capacitance_j_per_k=55.0,
)

MOTOR_SPECS = (
    LEG_MOTOR,
    LEG_MOTOR,
    LEG_MOTOR,
    LEG_MOTOR,
    WHEEL_MOTOR,
    WHEEL_MOTOR,
    GIMBAL_MOTOR,
    GIMBAL_MOTOR,
    FLYWHEEL_MOTOR,
    FLYWHEEL_MOTOR,
    FEEDER_MOTOR,
)

JOINT_TORQUE_LIMIT = np.asarray(
    [spec.joint_torque_limit_nm for spec in MOTOR_SPECS], dtype=np.float64
)
JOINT_SPEED_LIMIT = np.asarray(
    [spec.joint_speed_limit_rad_s for spec in MOTOR_SPECS], dtype=np.float64
)
REFLECTED_ARMATURE = np.asarray(
    [spec.reflected_armature for spec in MOTOR_SPECS], dtype=np.float64
)
VISCOUS_FRICTION = np.asarray(
    [spec.viscous_friction_nm_per_rad_s for spec in MOTOR_SPECS],
    dtype=np.float64,
)


def contract_dict() -> dict[str, object]:
    return {
        "robot": "actuatex_sentinel_11dof",
        "all_joint_names": list(ALL_JOINT_NAMES),
        "locomotion_joint_names": list(LOCOMOTION_JOINT_NAMES),
        "urdf_declaration_joint_names": list(URDF_DECLARATION_JOINT_NAMES),
        "policy_from_urdf": POLICY_FROM_URDF.tolist(),
        "sim_dt": SIM_DT,
        "policy_decimation": POLICY_DECIMATION,
        "policy_dt": POLICY_DT,
        "default_joint_position": SENTINEL_DEFAULT_JOINT_POSITION.tolist(),
        "motors": [asdict(spec) for spec in MOTOR_SPECS],
        "rmuc_2026": asdict(RMUC_2026),
    }


def contract_sha256() -> str:
    payload = json.dumps(
        contract_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def urdf_to_policy(values: np.ndarray) -> np.ndarray:
    """Reorder a URDF-declaration array into the public policy order."""

    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != FULL_ACTUATOR_DIM:
        raise ValueError(
            f"values must have last dimension {FULL_ACTUATOR_DIM}, got {array.shape}"
        )
    return np.take(array, POLICY_FROM_URDF, axis=-1)


def policy_to_urdf(values: np.ndarray) -> np.ndarray:
    """Reorder a public-policy array into URDF declaration order."""

    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != FULL_ACTUATOR_DIM:
        raise ValueError(
            f"values must have last dimension {FULL_ACTUATOR_DIM}, got {array.shape}"
        )
    return np.take(array, URDF_FROM_POLICY, axis=-1)


def validate_contract() -> None:
    if len(set(ALL_JOINT_NAMES)) != FULL_ACTUATOR_DIM:
        raise AssertionError("Sentinel joint names are not unique")
    if set(URDF_DECLARATION_JOINT_NAMES) != set(ALL_JOINT_NAMES):
        raise AssertionError("Sentinel URDF/policy joint-name sets drifted")
    if SENTINEL_DEFAULT_JOINT_POSITION.shape != (FULL_ACTUATOR_DIM,):
        raise AssertionError("Sentinel default-pose shape drifted")
    if len(MOTOR_SPECS) != FULL_ACTUATOR_DIM:
        raise AssertionError("Sentinel motor contract length drifted")
    for spec in MOTOR_SPECS:
        positive = (
            spec.gear_ratio,
            spec.torque_constant_nm_per_a,
            spec.back_emf_v_per_rad_s,
            spec.resistance_ohm,
            spec.current_limit_a,
            spec.joint_torque_limit_nm,
            spec.joint_speed_limit_rad_s,
            spec.transmission_efficiency,
            spec.thermal_resistance_k_per_w,
            spec.thermal_capacitance_j_per_k,
        )
        if not all(value > 0.0 for value in positive):
            raise AssertionError(f"invalid motor specification: {spec.name}")
        if not spec.derate_temperature_c < spec.shutdown_temperature_c:
            raise AssertionError(f"invalid motor thermal limits: {spec.name}")


validate_contract()
