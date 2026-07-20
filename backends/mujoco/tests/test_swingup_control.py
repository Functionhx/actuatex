from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.swingup_control import (  # noqa: E402
    EnergySwingupController,
    HybridEnergyLQRController,
    single_pole_energy,
    upright_target_energy,
)


def test_upright_has_target_energy_and_downward_is_lower() -> None:
    stationary = np.zeros(1)
    upright = single_pole_energy(np.zeros(1), stationary)[0]
    downward = single_pole_energy(np.full(1, np.pi), stationary)[0]
    assert np.isclose(upright, upright_target_energy())
    assert downward < upright


def test_energy_law_pumps_when_below_target() -> None:
    controller = EnergySwingupController(kick_force_n=0.0)
    controller.reset(1)
    state = np.array([[0.0, 2.0, 0.0, 1.0]])
    energy_error = (
        single_pole_energy(state[:, 1], state[:, 3])[0] - upright_target_energy()
    )
    action = controller.act(state)[0]
    assert energy_error < 0.0
    assert action > 0.0


def test_hybrid_controller_has_capture_hysteresis() -> None:
    controller = HybridEnergyLQRController(np.ones((1, 4)))
    controller.reset(1)
    controller.act(np.array([[0.0, 0.1, 0.0, 0.2]]))
    assert controller.balance_mode[0]
    controller.act(np.array([[0.0, 0.4, 0.0, 0.2]]))
    assert controller.balance_mode[0]
    controller.act(np.array([[0.0, 0.7, 0.0, 0.2]]))
    assert not controller.balance_mode[0]
