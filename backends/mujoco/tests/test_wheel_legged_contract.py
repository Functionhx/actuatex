"""Regression tests for the serial wheel-legged MuJoCo twin."""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


MUJOCO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MUJOCO_ROOT.parents[1]
sys.path.insert(0, str(MUJOCO_ROOT))

from wheel_legged_contract import (  # noqa: E402
    DEFAULT_JOINT_POSITION,
    POLICY_JOINT_NAMES,
    action_to_targets,
    build_observation,
    compute_mixed_pd_torque,
    projected_gravity,
)


MJCF_PATH = (
    REPO_ROOT / "robots" / "wheel_legged" / "mjcf" / "actuatex_serial_wheel_legged.xml"
)
URDF_PATH = (
    REPO_ROOT / "robots" / "wheel_legged" / "urdf" / "actuatex_serial_wheel_legged.urdf"
)


def object_name(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int
) -> str:
    value = mujoco.mj_id2name(model, object_type, object_id)
    assert value is not None
    return value


def test_policy_observation_and_action_contract() -> None:
    observation = build_observation(
        np.zeros(3),
        np.zeros(3),
        projected_gravity(np.array([1.0, 0.0, 0.0, 0.0])),
        np.array([0.5, 0.0, -0.3]),
        DEFAULT_JOINT_POSITION,
        np.zeros(6),
        np.zeros(6),
    )
    assert observation.shape == (28,)
    np.testing.assert_allclose(observation[:3], 0.0)
    np.testing.assert_allclose(observation[6:9], (0.0, 0.0, -1.0))
    np.testing.assert_allclose(observation[9:12], (0.5, 0.0, -0.3))
    np.testing.assert_allclose(observation[12:16], 0.0)

    leg_target, wheel_target = action_to_targets(np.ones(6))
    np.testing.assert_allclose(leg_target, (0.8, -0.25, 0.8, -0.25))
    np.testing.assert_allclose(wheel_target, (20.0, 20.0))


def test_mixed_pd_effort_and_clipping() -> None:
    torque = compute_mixed_pd_torque(
        DEFAULT_JOINT_POSITION,
        np.zeros(6),
        DEFAULT_JOINT_POSITION[:4] + 1.0,
        np.array((100.0, -100.0)),
    )
    np.testing.assert_allclose(torque, (30.0, 30.0, 30.0, 30.0, 12.0, -12.0))


def test_mjcf_dimensions_joint_and_actuator_order() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    assert (model.nq, model.nv, model.nu) == (13, 12, 6)
    assert model.nbody - 1 == 7
    assert np.isclose(model.body_mass.sum(), 12.28)
    assert np.isclose(model.opt.timestep, 0.005)

    actuator_joint_names = tuple(
        object_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator_id, 0]),
        )
        for actuator_id in range(model.nu)
    )
    assert actuator_joint_names == POLICY_JOINT_NAMES

    joint_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in POLICY_JOINT_NAMES
        ]
    )
    dof_addresses = model.jnt_dofadr[joint_ids]
    np.testing.assert_allclose(
        model.dof_armature[dof_addresses], (0.01,) * 4 + (0.02,) * 2
    )
    np.testing.assert_allclose(
        model.dof_frictionloss[dof_addresses], (0.02,) * 4 + (0.0,) * 2
    )


def test_mjcf_inertials_match_the_urdf() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    urdf_root = ET.parse(URDF_PATH).getroot()
    expected = {}
    for link in urdf_root.findall("link"):
        inertial = link.find("inertial")
        assert inertial is not None
        mass = float(inertial.find("mass").attrib["value"])
        inertia = inertial.find("inertia").attrib
        diagonal = np.array([float(inertia[axis]) for axis in ("ixx", "iyy", "izz")])
        expected[link.attrib["name"]] = (mass, diagonal)

    assert set(expected) == {
        object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    }
    for body_name, (expected_mass, expected_inertia) in expected.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert np.isclose(model.body_mass[body_id], expected_mass)
        np.testing.assert_allclose(model.body_inertia[body_id], expected_inertia)


def test_nominal_pd_dynamics_remain_finite_for_one_second() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    joint_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in POLICY_JOINT_NAMES
        ]
    )
    qpos_addresses = model.jnt_qposadr[joint_ids]
    dof_addresses = model.jnt_dofadr[joint_ids]
    data.qpos[:3] = (0.0, 0.0, 0.50)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[qpos_addresses] = DEFAULT_JOINT_POSITION
    mujoco.mj_forward(model, data)

    for _ in range(200):
        data.ctrl[:] = compute_mixed_pd_torque(
            data.qpos[qpos_addresses],
            data.qvel[dof_addresses],
            DEFAULT_JOINT_POSITION[:4],
            np.zeros(2),
        )
        mujoco.mj_step(model, data)

    # A two-wheeled body is intentionally not statically stable: the learned
    # policy, rather than the leg PD loop alone, supplies pitch balance.  This
    # smoke test therefore checks numerical integrity and actuator limits, not
    # a physically incorrect passive-standing assumption.
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert np.isfinite(data.actuator_force).all()
    np.testing.assert_array_less(
        np.abs(data.actuator_force), (30.0001,) * 4 + (12.0001,) * 2
    )
