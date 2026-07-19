import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MUJOCO_ROOT = REPO_ROOT / "backends/mujoco"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MUJOCO_ROOT))

from sentinel_env import MjSentinelEnv  # noqa: E402

from tasks.robomaster.contract import (  # noqa: E402
    ACTION_DIM,
    ALL_JOINT_NAMES,
    FULL_ACTUATOR_DIM,
    JOINT_TORQUE_LIMIT,
    JOINT_SPEED_LIMIT,
    MOTOR_SPECS,
    REFLECTED_ARMATURE,
    RMUC_2026,
    SENTINEL_DEFAULT_JOINT_POSITION,
    SIM_DT,
    URDF_DECLARATION_JOINT_NAMES,
    contract_sha256,
    policy_to_urdf,
    urdf_to_policy,
    validate_contract,
)

MJCF_PATH = REPO_ROOT / "robots/robomaster/mjcf/actuatex_sentinel.xml"
URDF_PATH = REPO_ROOT / "robots/robomaster/urdf/actuatex_sentinel.urdf"


def _object_name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_id: int,
) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    assert name is not None
    return name
from tasks.robomaster.launcher import (  # noqa: E402
    BallisticConfig,
    FireRequest,
    FrictionWheelLauncher,
    LauncherPhase,
    integrate_ballistic_rk4,
)
from tasks.robomaster.locomotion import (  # noqa: E402
    JOINT_DAMPING,
    JOINT_STIFFNESS,
    OBSERVATION_DIM,
    action_to_joint_targets,
    build_observation,
    compute_requested_joint_torque,
    projected_gravity,
)
from tasks.robomaster.powertrain import (  # noqa: E402
    RefereePowerMonitor,
    allocate_motor_power,
    integrate_motor_temperature,
    step_motor_bank,
    temperature_current_scale,
)
from tasks.robomaster.policy import load_policy  # noqa: E402
from tasks.robomaster.torch_powertrain import (  # noqa: E402
    TorchRefereePowerMonitor,
    allocate_motor_power_torch,
    integrate_motor_temperature_torch,
    step_motor_bank_torch,
)


def _fire_request(*, trigger: bool, armed: bool = True) -> FireRequest:
    return FireRequest(
        trigger=trigger,
        armed=armed,
        referee_enabled=True,
        safety_clear=True,
        gimbal_error_rad=0.0,
        requested_launch_speed_mps=24.0,
    )


def test_official_rmuc_constants_and_contract_are_pinned():
    validate_contract()
    assert FULL_ACTUATOR_DIM == 11
    assert len(contract_sha256()) == 64

    config_path = (
        REPO_ROOT / "robots/robomaster/config/rmuc_2026_v1.4.2.json"
    )
    pinned = json.loads(config_path.read_text(encoding="utf-8"))
    official = pinned["sentry_full_auto"]
    assert official["projectile_mass_kg"] == RMUC_2026.projectile_mass_kg
    assert (
        official["initial_launch_speed_limit_mps"]
        == RMUC_2026.initial_launch_speed_limit_mps
    )
    assert (
        official["chassis_power_limit_w"]
        == RMUC_2026.full_auto_chassis_power_limit_w
    )
    assert (
        official["chassis_buffer_energy_limit_j"]
        == RMUC_2026.chassis_buffer_energy_limit_j
    )


def test_mjcf_compiles_with_contract_order_limits_armatures_and_sensors():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    assert (model.nq, model.nv, model.nu) == (18, 17, FULL_ACTUATOR_DIM)
    assert model.nsensor == 44
    assert math.isclose(model.opt.timestep, SIM_DT)
    assert math.isclose(model.body_mass.sum(), 17.015, abs_tol=1.0e-12)

    actuator_joint_names = tuple(
        _object_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator_id, 0]),
        )
        for actuator_id in range(model.nu)
    )
    assert actuator_joint_names == ALL_JOINT_NAMES

    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ALL_JOINT_NAMES
        ]
    )
    dof_addresses = model.jnt_dofadr[joint_ids]
    np.testing.assert_allclose(model.dof_armature[dof_addresses], REFLECTED_ARMATURE)
    np.testing.assert_allclose(model.dof_damping[dof_addresses], 0.0)
    np.testing.assert_allclose(model.dof_frictionloss[dof_addresses], 0.0)
    np.testing.assert_allclose(
        model.actuator_ctrlrange,
        np.column_stack((-JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)),
    )

    sensor_names = {
        _object_name(model, mujoco.mjtObj.mjOBJ_SENSOR, index)
        for index in range(model.nsensor)
    }
    assert {
        "base_orientation",
        "base_angular_velocity",
        "base_linear_acceleration",
        "muzzle_position",
        "mid360_position",
        "front_armor_hit",
        "rear_armor_hit",
        "left_armor_hit",
        "right_armor_hit",
    } <= sensor_names


def test_mjcf_inertials_and_actuated_joint_order_match_urdf():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    urdf_root = ET.parse(URDF_PATH).getroot()
    expected_inertials = {}
    for link in urdf_root.findall("link"):
        inertial = link.find("inertial")
        assert inertial is not None
        mass = float(inertial.find("mass").attrib["value"])
        inertia = inertial.find("inertia").attrib
        diagonal = np.asarray(
            [float(inertia[axis]) for axis in ("ixx", "iyy", "izz")]
        )
        assert all(float(inertia[axis]) == 0.0 for axis in ("ixy", "ixz", "iyz"))
        expected_inertials[link.attrib["name"]] = (mass, diagonal)

    mjcf_bodies = {
        _object_name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(1, model.nbody)
    }
    assert set(expected_inertials) == mjcf_bodies
    for body_name, (mass, inertia) in expected_inertials.items():
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        assert math.isclose(model.body_mass[body_id], mass)
        np.testing.assert_allclose(model.body_inertia[body_id], inertia)

    urdf_actuated_joints = tuple(
        joint.attrib["name"]
        for joint in urdf_root.findall("joint")
        if joint.attrib["type"] != "fixed"
    )
    assert urdf_actuated_joints == URDF_DECLARATION_JOINT_NAMES
    assert set(urdf_actuated_joints) == set(ALL_JOINT_NAMES)

    urdf_values = np.arange(FULL_ACTUATOR_DIM)
    np.testing.assert_array_equal(
        policy_to_urdf(urdf_to_policy(urdf_values)), urdf_values
    )
    limits_by_name = {
        joint.attrib["name"]: joint.find("limit").attrib
        for joint in urdf_root.findall("joint")
        if joint.attrib["type"] != "fixed"
    }
    for index, joint_name in enumerate(ALL_JOINT_NAMES):
        assert math.isclose(
            float(limits_by_name[joint_name]["effort"]),
            JOINT_TORQUE_LIMIT[index],
        )
        assert math.isclose(
            float(limits_by_name[joint_name]["velocity"]),
            JOINT_SPEED_LIMIT[index],
        )


def test_power_limited_mjcf_dynamics_with_recoil_remains_finite():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ALL_JOINT_NAMES
        ]
    )
    qpos_addresses = model.jnt_qposadr[joint_ids]
    dof_addresses = model.jnt_dofadr[joint_ids]
    kp = np.asarray([80.0, 60.0, 80.0, 60.0, 0.0, 0.0, 20.0, 25.0, 0.0, 0.0, 0.0])
    kd = np.asarray([3.0, 2.5, 3.0, 2.5, 0.2, 0.2, 0.8, 0.8, 0.0, 0.0, 0.0])
    temperature = np.full(FULL_ACTUATOR_DIM, 25.0)
    maximum_power = 0.0
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    muzzle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "muzzle_site")

    for step in range(round(1.0 / SIM_DT)):
        position = data.qpos[qpos_addresses]
        velocity = data.qvel[dof_addresses]
        requested = kp * (SENTINEL_DEFAULT_JOINT_POSITION - position) - kd * velocity
        allocation = allocate_motor_power(
            requested,
            velocity,
            power_limit_w=RMUC_2026.full_auto_chassis_power_limit_w,
            temperature_c=temperature,
        )
        maximum_power = max(
            maximum_power, float(allocation.motor.positive_bus_power_w)
        )
        temperature = integrate_motor_temperature(
            temperature,
            allocation.motor.copper_loss_w,
            SIM_DT,
        )
        data.ctrl[:] = allocation.motor.joint_torque
        data.qfrc_applied[:] = 0.0
        if step == 250:
            recoil_impulse = np.asarray(
                [-RMUC_2026.projectile_mass_kg * 24.0, 0.0, 0.0]
            )
            mujoco.mj_applyFT(
                model,
                data,
                recoil_impulse / SIM_DT,
                np.zeros(3),
                data.site_xpos[muzzle_id],
                base_id,
                data.qfrc_applied,
            )
        mujoco.mj_step(model, data)

    assert maximum_power <= RMUC_2026.full_auto_chassis_power_limit_w + 1.0e-5
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert np.isfinite(data.actuator_force).all()
    assert np.isfinite(temperature).all()
    np.testing.assert_array_less(
        np.abs(data.actuator_force), JOINT_TORQUE_LIMIT + 1.0e-6
    )


def test_dc_motor_bank_enforces_voltage_current_speed_and_temperature_limits():
    requested = np.full(FULL_ACTUATOR_DIM, 1.0e6)
    stationary = step_motor_bank(requested, np.zeros(FULL_ACTUATOR_DIM))
    for index, spec in enumerate(MOTOR_SPECS):
        assert stationary.motor_current_a[index] <= spec.current_limit_a + 1.0e-12
        assert stationary.joint_torque[index] <= spec.joint_torque_limit_nm + 1.0e-12
        assert abs(stationary.terminal_voltage_v[index]) <= 24.0 + 1.0e-12

    at_no_load_speed = np.asarray(
        [spec.nominal_no_load_joint_speed for spec in MOTOR_SPECS]
    )
    fast = step_motor_bank(requested, at_no_load_speed)
    assert np.all(fast.joint_torque < stationary.joint_torque)

    temperatures = np.asarray(
        [spec.shutdown_temperature_c for spec in MOTOR_SPECS]
    )
    hot = step_motor_bank(requested, np.zeros(FULL_ACTUATOR_DIM), temperature_c=temperatures)
    np.testing.assert_allclose(hot.motor_current_a, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        [temperature_current_scale(spec.derate_temperature_c, spec) for spec in MOTOR_SPECS],
        1.0,
    )


def test_power_allocator_meets_scalar_and_batched_limits():
    requested = np.full((3, FULL_ACTUATOR_DIM), 100.0)
    velocity = np.broadcast_to(
        np.linspace(-10.0, 10.0, FULL_ACTUATOR_DIM), requested.shape
    )
    limits = np.asarray([0.0, 60.0, 100.0])
    allocation = allocate_motor_power(
        requested,
        velocity,
        power_limit_w=limits,
    )
    assert allocation.torque_scale.shape == (3,)
    assert np.all(allocation.motor.positive_bus_power_w <= limits + 1.0e-5)
    assert allocation.torque_scale[0] == 0.0
    assert allocation.torque_scale[1] < allocation.torque_scale[2] < 1.0


def test_power_allocator_can_limit_chassis_without_throttling_launcher():
    mask = np.asarray([True] * ACTION_DIM + [False] * 5)
    requested = np.zeros(FULL_ACTUATOR_DIM)
    requested[:ACTION_DIM] = 20.0
    requested[8:10] = (0.30, -0.30)
    allocation = allocate_motor_power(
        requested,
        np.zeros(FULL_ACTUATOR_DIM),
        power_limit_w=0.0,
        power_mask=mask,
    )
    assert allocation.torque_scale == 0.0
    assert allocation.accounted_bus_power_w == 0.0
    np.testing.assert_allclose(allocation.motor.joint_torque[:ACTION_DIM], 0.0)
    assert allocation.motor.joint_torque[8] > 0.0
    assert allocation.motor.joint_torque[9] < 0.0


def test_torch_motor_and_allocator_match_numpy_reference():
    rng = np.random.default_rng(17)
    requested = rng.uniform(-35.0, 35.0, size=(4, FULL_ACTUATOR_DIM))
    velocity = rng.uniform(-25.0, 25.0, size=(4, FULL_ACTUATOR_DIM))
    temperature = rng.uniform(25.0, 105.0, size=(4, FULL_ACTUATOR_DIM))
    limits = np.asarray([0.0, 60.0, 100.0, 300.0])

    numpy_motor = step_motor_bank(
        requested,
        velocity,
        temperature_c=temperature,
    )
    torch_motor = step_motor_bank_torch(
        torch.as_tensor(requested, dtype=torch.float64),
        torch.as_tensor(velocity, dtype=torch.float64),
        temperature_c=torch.as_tensor(temperature, dtype=torch.float64),
    )
    for field in (
        "joint_torque",
        "motor_current_a",
        "terminal_voltage_v",
        "electrical_power_w",
        "mechanical_power_w",
        "copper_loss_w",
        "current_limit_a",
    ):
        np.testing.assert_allclose(
            getattr(torch_motor, field).numpy(),
            getattr(numpy_motor, field),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    numpy_allocation = allocate_motor_power(
        requested,
        velocity,
        power_limit_w=limits,
        temperature_c=temperature,
        iterations=36,
    )
    torch_allocation = allocate_motor_power_torch(
        torch.as_tensor(requested, dtype=torch.float64),
        torch.as_tensor(velocity, dtype=torch.float64),
        power_limit_w=torch.as_tensor(limits, dtype=torch.float64),
        temperature_c=torch.as_tensor(temperature, dtype=torch.float64),
        iterations=36,
    )
    np.testing.assert_allclose(
        torch_allocation.torque_scale.numpy(),
        numpy_allocation.torque_scale,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        torch_allocation.motor.joint_torque.numpy(),
        numpy_allocation.motor.joint_torque,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    torch_heated = integrate_motor_temperature_torch(
        torch.as_tensor(temperature, dtype=torch.float64),
        torch_motor.copper_loss_w,
        0.002,
    )
    numpy_heated = integrate_motor_temperature(
        temperature,
        numpy_motor.copper_loss_w,
        0.002,
    )
    np.testing.assert_allclose(torch_heated.numpy(), numpy_heated, atol=1.0e-12)

    power_mask = np.asarray([True] * ACTION_DIM + [False] * 5)
    numpy_masked = allocate_motor_power(
        requested,
        velocity,
        power_limit_w=limits,
        power_mask=power_mask,
        temperature_c=temperature,
        iterations=36,
    )
    torch_masked = allocate_motor_power_torch(
        torch.as_tensor(requested, dtype=torch.float64),
        torch.as_tensor(velocity, dtype=torch.float64),
        power_limit_w=torch.as_tensor(limits, dtype=torch.float64),
        power_mask=torch.as_tensor(power_mask),
        temperature_c=torch.as_tensor(temperature, dtype=torch.float64),
        iterations=36,
    )
    np.testing.assert_allclose(
        torch_masked.motor.joint_torque.numpy(),
        numpy_masked.motor.joint_torque,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        torch_masked.accounted_bus_power_w.numpy(),
        numpy_masked.accounted_bus_power_w,
        atol=1.0e-9,
    )


def test_torch_motor_model_maps_specs_by_joint_name_not_storage_index():
    names = tuple(reversed(ALL_JOINT_NAMES))
    requested = torch.linspace(-20.0, 20.0, FULL_ACTUATOR_DIM, dtype=torch.float64)
    velocity = torch.linspace(-5.0, 5.0, FULL_ACTUATOR_DIM, dtype=torch.float64)
    reference = step_motor_bank_torch(requested, velocity)
    reversed_result = step_motor_bank_torch(
        requested.flip(-1),
        velocity.flip(-1),
        joint_names=names,
    )
    torch.testing.assert_close(
        reversed_result.joint_torque.flip(-1), reference.joint_torque
    )


def test_motor_thermal_state_heats_derates_and_cools():
    temperature = np.full(FULL_ACTUATOR_DIM, 25.0)
    loss = np.full(FULL_ACTUATOR_DIM, 30.0)
    heated = temperature.copy()
    for _ in range(1000):
        heated = integrate_motor_temperature(heated, loss, 0.01)
    assert np.all(heated > temperature)

    cooled = heated.copy()
    for _ in range(1000):
        cooled = integrate_motor_temperature(cooled, np.zeros(FULL_ACTUATOR_DIM), 0.01)
    assert np.all(cooled < heated)
    assert np.all(cooled >= 25.0)


def test_referee_power_monitor_consumes_60_j_then_cuts_power_for_5_seconds():
    monitor = RefereePowerMonitor()
    for _ in range(299):
        monitor.step(200.0, 0.002)
        assert monitor.chassis_enabled
    state = monitor.step(200.0, 0.002)
    assert state.buffer_energy_j == 0.0
    assert not monitor.chassis_enabled
    assert math.isclose(state.powered_off_remaining_s, 5.0, abs_tol=1.0e-12)

    for _ in range(2499):
        monitor.step(200.0, 0.002)
        assert not monitor.chassis_enabled
    monitor.step(200.0, 0.002)
    assert monitor.chassis_enabled
    assert monitor.state.buffer_energy_j == 60.0


def test_torch_referee_monitor_matches_numpy_cutoff_and_recovery():
    numpy_monitor = RefereePowerMonitor()
    torch_monitor = TorchRefereePowerMonitor(
        2,
        device="cpu",
        dtype=torch.float64,
    )
    for step in range(2800):
        numpy_state = numpy_monitor.step(200.0, 0.002)
        torch_monitor.step(torch.tensor([200.0, 50.0], dtype=torch.float64), 0.002)
        assert bool(torch_monitor.chassis_enabled[0]) == numpy_monitor.chassis_enabled
        assert math.isclose(
            float(torch_monitor.buffer_energy_j[0]),
            numpy_state.buffer_energy_j,
            abs_tol=1.0e-9,
        )
        if step >= 49:
            assert float(torch_monitor.buffer_energy_j[1]) == 60.0


def test_launcher_spins_up_feeds_fires_recoils_and_obeys_heat_gate():
    launcher = FrictionWheelLauncher()
    launcher.reset(ammunition=2)
    launch = None
    for index in range(3000):
        result = launcher.step(_fire_request(trigger=index >= 800), 0.002)
        if result.launch is not None:
            launch = result.launch
            break

    assert launch is not None
    assert 0.0 < launch.speed_mps <= RMUC_2026.initial_launch_speed_limit_mps
    assert math.isclose(
        launch.kinetic_energy_j,
        0.5 * RMUC_2026.projectile_mass_kg * launch.speed_mps**2,
        rel_tol=1.0e-12,
    )
    np.testing.assert_allclose(
        launch.recoil_impulse_body_ns,
        [-RMUC_2026.projectile_mass_kg * launch.speed_mps, 0.0, 0.0],
    )
    assert launch.ammunition_after == 1
    assert launch.barrel_heat_after == RMUC_2026.barrel_heat_per_projectile
    assert launcher.state.phase == LauncherPhase.COOLDOWN

    launcher.state = launcher.state.__class__(
        **{
            **launcher.state.__dict__,
            "phase": LauncherPhase.READY,
            "barrel_heat": RMUC_2026.full_auto_barrel_heat_limit,
            "cooldown_remaining_s": 0.0,
        }
    )
    rejected = launcher.step(_fire_request(trigger=True), 0.002)
    assert rejected.launch is None
    assert "barrel_heat" in rejected.rejection_reasons


def test_launcher_disarm_and_safe_stop_fail_closed():
    launcher = FrictionWheelLauncher()
    launcher.reset(ammunition=5)
    result = launcher.step(_fire_request(trigger=True, armed=False), 0.002)
    assert result.launch is None
    assert result.state.phase == LauncherPhase.DISARMED
    assert "disarmed" in result.rejection_reasons

    launcher.safe_stop("watchdog")
    stopped = launcher.step(_fire_request(trigger=True), 0.002)
    assert stopped.launch is None
    assert stopped.state.phase == LauncherPhase.SAFE_STOP
    assert stopped.rejection_reasons == ("watchdog",)


def test_ballistics_matches_vacuum_solution_and_drag_reduces_range():
    no_drag = BallisticConfig(air_density_kg_m3=0.0)
    position = np.asarray([0.0, 0.0, 1.0])
    velocity = np.asarray([24.0, 0.0, 4.0])
    dt = 0.001
    vacuum_position = position.copy()
    vacuum_velocity = velocity.copy()
    drag_position = position.copy()
    drag_velocity = velocity.copy()
    for _ in range(1000):
        vacuum_position, vacuum_velocity = integrate_ballistic_rk4(
            vacuum_position, vacuum_velocity, dt, no_drag
        )
        drag_position, drag_velocity = integrate_ballistic_rk4(
            drag_position, drag_velocity, dt
        )

    np.testing.assert_allclose(vacuum_position, [24.0, 0.0, 0.095], atol=1.0e-9)
    np.testing.assert_allclose(vacuum_velocity, [24.0, 0.0, -5.81], atol=1.0e-9)
    assert drag_position[0] < vacuum_position[0]
    assert drag_velocity[0] < vacuum_velocity[0]


def test_locomotion_action_observation_and_pd_contract():
    action = np.asarray([1.0, -1.0, 0.5, -0.5, 1.0, -1.0])
    position_target, velocity_target = action_to_joint_targets(action)
    np.testing.assert_allclose(
        position_target[:4],
        SENTINEL_DEFAULT_JOINT_POSITION[:4] + 0.45 * action[:4],
    )
    np.testing.assert_allclose(velocity_target[4:6], [20.0, -20.0])
    requested = compute_requested_joint_torque(
        SENTINEL_DEFAULT_JOINT_POSITION,
        np.zeros(FULL_ACTUATOR_DIM),
        action,
    )
    np.testing.assert_allclose(
        requested,
        JOINT_STIFFNESS
        * (position_target - SENTINEL_DEFAULT_JOINT_POSITION)
        + JOINT_DAMPING * velocity_target,
    )
    np.testing.assert_allclose(projected_gravity([1.0, 0.0, 0.0, 0.0]), [0, 0, -1])
    observation = build_observation(
        np.zeros(3),
        np.zeros(3),
        np.asarray([0.0, 0.0, -1.0]),
        np.asarray([0.5, 0.0, -0.3]),
        SENTINEL_DEFAULT_JOINT_POSITION,
        np.zeros(FULL_ACTUATOR_DIM),
        action,
    )
    assert observation.shape == (OBSERVATION_DIM,)
    np.testing.assert_allclose(observation[-ACTION_DIM:], action)


def _test_ppo_state(prefix: str) -> tuple[dict[str, torch.Tensor], torch.nn.Module]:
    torch.manual_seed(29)
    actor = torch.nn.Sequential(
        torch.nn.Linear(OBSERVATION_DIM, 8),
        torch.nn.ELU(),
        torch.nn.Linear(8, ACTION_DIM),
    )
    state = {f"{prefix}{key}": value for key, value in actor.state_dict().items()}
    return state, actor


def test_policy_loader_supports_both_rsl_rl_ppo_checkpoint_layouts():
    observation = torch.linspace(-1.0, 1.0, OBSERVATION_DIM).unsqueeze(0)
    old_state, old_reference = _test_ppo_state("actor.")
    old = load_policy(
        {
            "model_state_dict": old_state,
            "infos": {"backend": "MuJoCo 3.10"},
        }
    )
    assert old.algorithm == "ppo"
    assert old.checkpoint_format == "rsl_rl_1_actor_critic"
    assert old.metadata["backend"] == "MuJoCo 3.10"
    torch.testing.assert_close(old.actor(observation), old_reference(observation))

    new_state, new_reference = _test_ppo_state("mlp.")
    new = load_policy(
        {
            "actor_state_dict": new_state,
            "infos": {"backend": "Isaac Sim 6.0.1 GA"},
        }
    )
    assert new.algorithm == "ppo"
    assert new.checkpoint_format == "rsl_rl_5_split_actor"
    assert new.metadata["backend"] == "Isaac Sim 6.0.1 GA"
    torch.testing.assert_close(new.actor(observation), new_reference(observation))


def test_clean_mujoco_reset_is_deterministic():
    env = MjSentinelEnv(
        num_envs=2,
        num_threads=1,
        seed=31,
        add_noise=False,
        randomize_reset=False,
    )
    try:
        env.set_command(np.asarray([0.5, 0.0, 0.0]))
        first, _ = env.reset()
        first_state = env.diagnostics()
        env.step(torch.ones((2, ACTION_DIM)))
        second, _ = env.reset()
        second_state = env.diagnostics()
        torch.testing.assert_close(first, second)
        for key in (
            "base_position",
            "base_linear_velocity_body",
            "base_angular_velocity_body",
            "joint_position",
            "joint_velocity",
        ):
            np.testing.assert_allclose(first_state[key], second_state[key])
    finally:
        env.close()


def test_vectorized_mujoco_sentinel_environment_smoke():
    env = MjSentinelEnv(
        num_envs=4,
        num_threads=2,
        seed=23,
        add_noise=False,
        maximum_command_delay_steps=10,
    )
    try:
        observation, _ = env.reset()
        assert observation.shape == (4, OBSERVATION_DIM)
        assert np.all(env.command_delay_steps <= 10)
        env.set_command(np.asarray([0.5, 0.0, 0.0]))
        for _ in range(10):
            observation, _, reward, done, _ = env.step(
                torch.zeros((4, ACTION_DIM))
            )
        assert bool(torch.isfinite(observation).all())
        assert bool(torch.isfinite(reward).all())
        assert done.shape == (4,)
        assert np.all(env.last_chassis_power_w <= 180.0 + 1.0e-4)
        assert np.all(env.motor_temperature_c >= 25.0)
    finally:
        env.close()
