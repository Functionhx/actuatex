from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.classical_control import (  # noqa: E402
    CascadedPIDController,
    LQGController,
    PIDController,
    StateFeedbackController,
)
from tasks.inverted_pendulum.robust_control import (  # noqa: E402
    CollocatedFeedbackLinearizationController,
    DiscreteSlidingModeController,
    PartialFeedbackLinearizationController,
)
from tasks.inverted_pendulum.state_estimators import (  # noqa: E402
    ComplementaryFilterLQRController,
    ExtendedKalmanLQRController,
    LinearObserverLQRController,
    nonlinear_single_cartpole_step as numpy_nonlinear_step,
)
from tasks.inverted_pendulum.torch_control import (  # noqa: E402
    TorchCascadedPIDController,
    TorchCollocatedFeedbackLinearizationController,
    TorchComplementaryFilterLQRController,
    TorchDiscreteSlidingModeController,
    TorchExtendedKalmanLQRController,
    TorchLinearOutputFeedbackController,
    TorchPIDController,
    TorchPartialFeedbackLinearizationController,
    TorchStateFeedbackController,
    nonlinear_single_cartpole_step as torch_nonlinear_step,
)


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _model():
    matrix_a = np.array(
        [
            [1.0, 0.0, 0.02, 0.0],
            [0.0, 1.01, 0.0, 0.02],
            [0.0, 0.0, 0.99, 0.0],
            [0.0, 0.4, 0.0, 1.0],
        ]
    )
    matrix_b = np.array([[0.001], [-0.002], [0.02], [-0.04]])
    measurement_c = np.zeros((2, 4))
    measurement_c[:, :2] = np.eye(2)
    feedback_gain = np.array([[-1.0, -28.0, -2.0, -5.5]])
    correction_gain = np.array(
        [[0.7, 0.0], [0.0, 0.7], [0.4, 0.05], [0.05, 0.6]]
    )
    return matrix_a, matrix_b, measurement_c, feedback_gain, correction_gain


def _states() -> np.ndarray:
    return np.array(
        [
            [0.12, 0.08, -0.04, 0.03],
            [-0.19, -0.12, 0.07, -0.08],
            [0.03, 0.20, 0.02, 0.12],
        ]
    )


def _assert_actions_equal(reference, tensor) -> None:
    np.testing.assert_allclose(
        tensor.detach().cpu().numpy(), reference, rtol=1.0e-8, atol=1.0e-9
    )


def test_torch_state_feedback_pid_and_cascade_match_numpy() -> None:
    state = _states()
    _, _, _, gain, _ = _model()
    reference = StateFeedbackController(gain)
    tensor = TorchStateFeedbackController(
        gain, device=DEVICE, dtype=DTYPE
    )
    _assert_actions_equal(reference.act(state), tensor.act(torch.from_numpy(state)))

    pid_parameters = dict(
        cart_kp=1.2,
        cart_kd=0.8,
        angle_kp=26.0,
        angle_ki=0.1,
        angle_kd=5.0,
    )
    pid_reference = PIDController(**pid_parameters)
    pid_tensor = TorchPIDController(
        **pid_parameters, device=DEVICE, dtype=DTYPE
    )
    pid_reference.reset(len(state))
    pid_tensor.reset(len(state))
    for _ in range(3):
        _assert_actions_equal(
            pid_reference.act(state), pid_tensor.act(torch.from_numpy(state))
        )

    cascade_parameters = dict(
        outer_kp=0.08,
        outer_kd=0.12,
        inner_kp=26.0,
        inner_ki=0.05,
        inner_kd=5.0,
    )
    cascade_reference = CascadedPIDController(**cascade_parameters)
    cascade_tensor = TorchCascadedPIDController(
        **cascade_parameters, device=DEVICE, dtype=DTYPE
    )
    cascade_reference.reset(len(state))
    cascade_tensor.reset(len(state))
    for _ in range(3):
        _assert_actions_equal(
            cascade_reference.act(state),
            cascade_tensor.act(torch.from_numpy(state)),
        )


def test_torch_linear_observers_match_numpy() -> None:
    state = _states()
    matrix_a, matrix_b, measurement_c, gain, correction_gain = _model()
    measurement = state @ measurement_c.T
    for reference in (
        LQGController(
            matrix_a, matrix_b, measurement_c, gain, correction_gain
        ),
        LinearObserverLQRController(
            matrix_a, matrix_b, measurement_c, gain, correction_gain
        ),
    ):
        tensor = TorchLinearOutputFeedbackController(
            matrix_a,
            matrix_b,
            measurement_c,
            gain,
            correction_gain,
            device=DEVICE,
            dtype=DTYPE,
        )
        reference.reset(len(state))
        tensor.reset(len(state))
        for _ in range(4):
            _assert_actions_equal(
                reference.act_from_measurement(measurement),
                tensor.act_from_measurement(torch.from_numpy(measurement)),
            )


def test_torch_complementary_and_sliding_controllers_match_numpy() -> None:
    state = _states()
    matrix_a, matrix_b, measurement_c, gain, _ = _model()
    measurement = state @ measurement_c.T
    reference_filter = ComplementaryFilterLQRController(
        matrix_a, matrix_b, measurement_c, gain
    )
    tensor_filter = TorchComplementaryFilterLQRController(
        matrix_a,
        matrix_b,
        measurement_c,
        gain,
        device=DEVICE,
        dtype=DTYPE,
    )
    reference_filter.reset(len(state))
    tensor_filter.reset(len(state))
    for offset in (0.0, 0.002, -0.001):
        noisy_measurement = measurement + offset
        _assert_actions_equal(
            reference_filter.act_from_measurement(noisy_measurement),
            tensor_filter.act_from_measurement(
                torch.from_numpy(noisy_measurement)
            ),
        )

    reference_sliding = DiscreteSlidingModeController(
        matrix_a, matrix_b, gain
    )
    tensor_sliding = TorchDiscreteSlidingModeController(
        matrix_a, matrix_b, gain, device=DEVICE, dtype=DTYPE
    )
    _assert_actions_equal(
        reference_sliding.act(state), tensor_sliding.act(torch.from_numpy(state))
    )


def test_torch_feedback_linearization_controllers_match_numpy() -> None:
    state = _states()
    acceleration_gain = np.array([-0.8, -20.0, -1.5, -4.0])
    reference_collocated = CollocatedFeedbackLinearizationController(
        acceleration_gain
    )
    tensor_collocated = TorchCollocatedFeedbackLinearizationController(
        acceleration_gain, device=DEVICE, dtype=DTYPE
    )
    _assert_actions_equal(
        reference_collocated.act(state),
        tensor_collocated.act(torch.from_numpy(state)),
    )

    reference_partial = PartialFeedbackLinearizationController()
    tensor_partial = TorchPartialFeedbackLinearizationController(
        device=DEVICE, dtype=DTYPE
    )
    _assert_actions_equal(
        reference_partial.act(state),
        tensor_partial.act(torch.from_numpy(state)),
    )


def test_torch_nonlinear_model_and_ekf_match_numpy() -> None:
    state = _states()
    force = np.array([0.2, -0.4, 0.1])
    torch_step = torch_nonlinear_step(
        torch.from_numpy(state), torch.from_numpy(force)
    )
    np.testing.assert_allclose(
        torch_step.numpy(),
        numpy_nonlinear_step(state, force),
        rtol=1.0e-9,
        atol=1.0e-10,
    )

    _, _, measurement_c, gain, _ = _model()
    measurement = state @ measurement_c.T
    reference = ExtendedKalmanLQRController(
        gain, measurement_noise_std=0.002
    )
    tensor = TorchExtendedKalmanLQRController(
        gain,
        measurement_noise_std=0.002,
        device=DEVICE,
        dtype=DTYPE,
    )
    reference.reset(len(state))
    tensor.reset(len(state))
    for offset in (0.0, 0.001, -0.0005):
        noisy_measurement = measurement + offset
        _assert_actions_equal(
            reference.act_from_measurement(noisy_measurement),
            tensor.act_from_measurement(torch.from_numpy(noisy_measurement)),
        )
