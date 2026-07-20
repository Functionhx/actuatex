"""Regression tests for the shared serial inverted-pendulum benchmark."""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.contract import (  # noqa: E402
    OBSERVATION_DIM,
    absolute_pole_angles,
    build_observation,
    compute_reward,
    terminated,
)


def test_shared_observation_padding_and_presence_mask() -> None:
    for order in (1, 2, 3):
        angles = np.arange(order, dtype=np.float64).reshape(1, order) * 0.1
        velocities = np.zeros_like(angles)
        observation = build_observation(
            np.array([0.24]), np.array([0.5]), angles, velocities, order
        )
        assert observation.shape == (1, OBSERVATION_DIM)
        np.testing.assert_allclose(observation[0, 11 : 11 + order], 1.0)
        np.testing.assert_allclose(observation[0, 11 + order : 14], 0.0)
        for pole_index in range(order, 3):
            np.testing.assert_allclose(
                observation[0, 2 + 3 * pole_index : 5 + 3 * pole_index], 0.0
            )


def test_absolute_angles_follow_the_serial_chain() -> None:
    relative = np.array([[0.2, 0.3, -0.1]])
    np.testing.assert_allclose(absolute_pole_angles(relative), [[0.2, 0.5, 0.4]])


def test_reward_and_termination_use_world_upright_angles() -> None:
    cart = np.zeros(1)
    relative = np.array([[0.8, 0.8]])
    assert terminated(cart, relative).item()
    upright_reward = compute_reward(
        cart,
        cart,
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        cart,
        cart,
        np.array([False]),
    )
    bent_reward = compute_reward(
        cart,
        cart,
        relative,
        np.zeros((1, 2)),
        cart,
        cart,
        np.array([True]),
    )
    assert upright_reward.item() > bent_reward.item()


def test_mjcf_assets_have_one_input_and_expected_dofs() -> None:
    for order in (1, 2, 3):
        path = (
            REPO_ROOT
            / "robots"
            / "inverted_pendulum"
            / "mjcf"
            / f"actuatex_cartpole_{order}.xml"
        )
        model = mujoco.MjModel.from_xml_path(str(path))
        assert (model.nq, model.nv, model.nu) == (order + 1, order + 1, 1)
        assert np.isclose(model.body_mass.sum(), 1.0 + 0.2 * order)
        actuator_joint = int(model.actuator_trnid[0, 0])
        assert (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, actuator_joint)
            == "cart_slide"
        )
        np.testing.assert_allclose(model.actuator_ctrlrange[0], (-20.0, 20.0))


def test_urdf_and_mjcf_mass_and_diagonal_inertia_match() -> None:
    for order in (1, 2, 3):
        urdf_path = (
            REPO_ROOT
            / "robots"
            / "inverted_pendulum"
            / "urdf"
            / f"actuatex_cartpole_{order}.urdf"
        )
        mjcf_path = (
            REPO_ROOT
            / "robots"
            / "inverted_pendulum"
            / "mjcf"
            / f"actuatex_cartpole_{order}.xml"
        )
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        urdf_root = ET.parse(urdf_path).getroot()
        expected = {}
        for link in urdf_root.findall("link"):
            if link.attrib["name"] == "rail":
                continue
            inertial = link.find("inertial")
            assert inertial is not None
            inertia = inertial.find("inertia")
            assert inertia is not None
            expected[link.attrib["name"]] = (
                float(inertial.find("mass").attrib["value"]),
                np.array([float(inertia.attrib[key]) for key in ("ixx", "iyy", "izz")]),
            )

        for body_name, (mass, inertia) in expected.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            assert body_id >= 0
            assert np.isclose(model.body_mass[body_id], mass)
            np.testing.assert_allclose(model.body_inertia[body_id], inertia)
