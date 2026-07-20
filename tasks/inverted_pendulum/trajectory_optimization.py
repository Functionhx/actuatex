"""Direct trajectory optimization and TVLQR for the single cart-pole."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import ACTION_FORCE_SCALE_N, POLICY_DT
from .state_estimators import nonlinear_single_cartpole_step


@dataclass(frozen=True)
class OptimizedTrajectory:
    states: np.ndarray
    forces: np.ndarray
    objective: float
    solver_iterations: int
    return_status: str


def _rollout_forces(initial_state: np.ndarray, forces: np.ndarray) -> np.ndarray:
    forces = np.asarray(forces, dtype=np.float64).reshape(-1)
    states = np.empty((forces.size + 1, 4), dtype=np.float64)
    states[0] = np.asarray(initial_state, dtype=np.float64).reshape(4)
    for step, force in enumerate(forces):
        states[step + 1] = nonlinear_single_cartpole_step(
            states[step : step + 1], np.array([force])
        )[0]
    return states


def _swingup_cost(states: np.ndarray, forces: np.ndarray) -> float:
    running = (
        0.2 * np.square(states[:-1, 0])
        + 4.0 * (1.0 - np.cos(states[:-1, 1]))
        + 0.02 * np.square(states[:-1, 2])
        + 0.02 * np.square(states[:-1, 3])
        + 0.002 * np.square(forces)
    )
    terminal = (
        800.0 * states[-1, 0] ** 2
        + 1600.0 * (1.0 - np.cos(states[-1, 1]))
        + 100.0 * states[-1, 2] ** 2
        + 100.0 * states[-1, 3] ** 2
    )
    return float(POLICY_DT * np.sum(running) + terminal)


def iterative_lqr_swingup(
    initial_state: np.ndarray,
    initial_forces: np.ndarray,
    *,
    max_iterations: int = 100,
    convergence_tolerance: float = 1.0e-6,
) -> OptimizedTrajectory:
    """Optimize a bounded swing-up sequence with iterative LQR and line search."""

    initial_state = np.asarray(initial_state, dtype=np.float64).reshape(4)
    forces = np.clip(
        np.asarray(initial_forces, dtype=np.float64).reshape(-1),
        -ACTION_FORCE_SCALE_N,
        ACTION_FORCE_SCALE_N,
    )
    if forces.size == 0 or max_iterations <= 0:
        raise ValueError("iLQR needs a non-empty initial control sequence")
    states = _rollout_forces(initial_state, forces)
    cost = _swingup_cost(states, forces)
    regularization = 1.0e-4
    status = "maximum_iterations"
    completed_iterations = 0

    for iteration in range(max_iterations):
        completed_iterations = iteration + 1
        matrices_a, matrices_b = linearize_trajectory(states, forces)
        terminal_state = states[-1]
        value_gradient = np.array(
            [
                1600.0 * terminal_state[0],
                1600.0 * np.sin(terminal_state[1]),
                200.0 * terminal_state[2],
                200.0 * terminal_state[3],
            ]
        )
        value_hessian = np.diag(
            [
                1600.0,
                1600.0 * np.cos(terminal_state[1]),
                200.0,
                200.0,
            ]
        )
        feedforward = np.empty(forces.size, dtype=np.float64)
        feedback = np.empty((forces.size, 1, 4), dtype=np.float64)
        backward_ok = True
        for step in range(forces.size - 1, -1, -1):
            state = states[step]
            force = forces[step]
            cost_x = POLICY_DT * np.array(
                [
                    0.4 * state[0],
                    4.0 * np.sin(state[1]),
                    0.04 * state[2],
                    0.04 * state[3],
                ]
            )
            cost_xx = POLICY_DT * np.diag([0.4, 4.0 * np.cos(state[1]), 0.04, 0.04])
            cost_u = POLICY_DT * 0.004 * force
            cost_uu = POLICY_DT * 0.004
            matrix_a = matrices_a[step]
            matrix_b = matrices_b[step]
            q_x = cost_x + matrix_a.T @ value_gradient
            q_u = cost_u + float((matrix_b.T @ value_gradient).item())
            q_xx = cost_xx + matrix_a.T @ value_hessian @ matrix_a
            q_ux = matrix_b.T @ value_hessian @ matrix_a
            q_uu = cost_uu + float((matrix_b.T @ value_hessian @ matrix_b).item())
            regularized_q_uu = q_uu + regularization
            if not np.isfinite(regularized_q_uu) or regularized_q_uu <= 1.0e-10:
                backward_ok = False
                break
            feedforward[step] = -q_u / regularized_q_uu
            feedback[step, 0] = -q_ux[0] / regularized_q_uu
            gain = feedback[step]
            step_feedforward = feedforward[step]
            value_gradient = (
                q_x
                + gain.T[:, 0] * regularized_q_uu * step_feedforward
                + gain.T[:, 0] * q_u
                + q_ux.T[:, 0] * step_feedforward
            )
            value_hessian = (
                q_xx
                + gain.T @ (regularized_q_uu * gain)
                + gain.T @ q_ux
                + q_ux.T @ gain
            )
            value_hessian = 0.5 * (value_hessian + value_hessian.T)
        if not backward_ok:
            regularization *= 10.0
            if regularization > 1.0e10:
                status = "backward_pass_failed"
                break
            continue

        accepted = False
        previous_cost = cost
        for line_scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            candidate_states = np.empty_like(states)
            candidate_forces = np.empty_like(forces)
            candidate_states[0] = initial_state
            for step in range(forces.size):
                state_error = candidate_states[step] - states[step]
                state_error[1] = np.arctan2(
                    np.sin(state_error[1]), np.cos(state_error[1])
                )
                candidate_forces[step] = np.clip(
                    forces[step]
                    + line_scale * feedforward[step]
                    + float((feedback[step] @ state_error).item()),
                    -ACTION_FORCE_SCALE_N,
                    ACTION_FORCE_SCALE_N,
                )
                candidate_states[step + 1] = nonlinear_single_cartpole_step(
                    candidate_states[step : step + 1],
                    candidate_forces[step : step + 1],
                )[0]
            candidate_cost = _swingup_cost(candidate_states, candidate_forces)
            if candidate_cost < cost:
                states = candidate_states
                forces = candidate_forces
                cost = candidate_cost
                accepted = True
                regularization = max(regularization * 0.5, 1.0e-9)
                break
        if not accepted:
            regularization *= 10.0
            if regularization > 1.0e10:
                status = "line_search_failed"
                break
            continue
        if previous_cost - cost < convergence_tolerance:
            status = "converged"
            break
    return OptimizedTrajectory(
        states=states,
        forces=forces,
        objective=cost,
        solver_iterations=completed_iterations,
        return_status=status,
    )


def _casadi_rk4_step(state, force, sample_time: float):
    """CasADi form of the physical cart-pole RK4 transition."""

    import casadi as ca

    cart_mass = 1.0
    pole_mass = 0.20
    center_length = 0.30
    center_inertia = pole_mass * (0.04**2 + 0.60**2) / 12.0
    hinge_inertia = center_inertia + pole_mass * center_length**2

    def derivative(value):
        theta = value[1]
        angular_velocity = value[3]
        coupling = pole_mass * center_length * ca.cos(theta)
        rhs_cart = (
            force + pole_mass * center_length * ca.sin(theta) * angular_velocity**2
        )
        rhs_pole = pole_mass * 9.81 * center_length * ca.sin(theta)
        determinant = (cart_mass + pole_mass) * hinge_inertia - coupling**2
        cart_acceleration = (
            rhs_cart * hinge_inertia - coupling * rhs_pole
        ) / determinant
        angular_acceleration = (
            (cart_mass + pole_mass) * rhs_pole - coupling * rhs_cart
        ) / determinant
        return ca.vertcat(
            value[2],
            value[3],
            cart_acceleration,
            angular_acceleration,
        )

    half_time = 0.5 * sample_time
    k1 = derivative(state)
    k2 = derivative(state + half_time * k1)
    k3 = derivative(state + half_time * k2)
    k4 = derivative(state + sample_time * k3)
    return state + (sample_time / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_swingup_multiple_shooting(
    initial_state: np.ndarray,
    *,
    horizon: int,
    initial_states: np.ndarray | None = None,
    initial_forces: np.ndarray | None = None,
) -> OptimizedTrajectory:
    """Solve a bounded nonlinear swing-up with CasADi multiple shooting."""

    import casadi as ca

    initial_state = np.asarray(initial_state, dtype=np.float64).reshape(4)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    opti = ca.Opti()
    states = opti.variable(4, horizon + 1)
    forces = opti.variable(1, horizon)
    opti.subject_to(states[:, 0] == initial_state)
    for step in range(horizon):
        opti.subject_to(
            states[:, step + 1]
            == _casadi_rk4_step(states[:, step], forces[0, step], POLICY_DT)
        )
    opti.subject_to(opti.bounded(-2.2, states[0, :], 2.2))
    opti.subject_to(opti.bounded(-ACTION_FORCE_SCALE_N, forces, ACTION_FORCE_SCALE_N))
    opti.subject_to(opti.bounded(-0.06, states[0, -1], 0.06))
    opti.subject_to(opti.bounded(-0.08, states[1, -1], 0.08))
    opti.subject_to(opti.bounded(-0.12, states[2, -1], 0.12))
    opti.subject_to(opti.bounded(-0.15, states[3, -1], 0.15))

    running = 0
    for step in range(horizon):
        running += (
            0.2 * states[0, step] ** 2
            + 4.0 * (1.0 - ca.cos(states[1, step]))
            + 0.02 * states[2, step] ** 2
            + 0.02 * states[3, step] ** 2
            + 0.002 * forces[0, step] ** 2
        )
    terminal = (
        800.0 * states[0, -1] ** 2
        + 1600.0 * (1.0 - ca.cos(states[1, -1]))
        + 100.0 * states[2, -1] ** 2
        + 100.0 * states[3, -1] ** 2
    )
    opti.minimize(POLICY_DT * running + terminal)
    if initial_states is not None:
        guess = np.asarray(initial_states, dtype=np.float64)
        if guess.shape != (horizon + 1, 4):
            raise ValueError("initial state guess has the wrong shape")
        opti.set_initial(states, guess.T)
    else:
        theta_guess = np.linspace(initial_state[1], 0.0, horizon + 1)
        opti.set_initial(states[1, :], theta_guess)
    if initial_forces is not None:
        force_guess = np.asarray(initial_forces, dtype=np.float64).reshape(-1)
        if force_guess.shape != (horizon,):
            raise ValueError("initial force guess has the wrong shape")
        opti.set_initial(forces, force_guess[None, :])
    opti.solver(
        "ipopt",
        {"expand": True},
        {
            "max_iter": 2500,
            "tol": 1.0e-7,
            "acceptable_tol": 1.0e-5,
            "print_level": 0,
            "sb": "yes",
        },
    )
    solution = opti.solve()
    stats = solution.stats()
    return OptimizedTrajectory(
        states=np.asarray(solution.value(states)).T,
        forces=np.asarray(solution.value(forces)).reshape(-1),
        objective=float(solution.value(opti.f)),
        solver_iterations=int(stats.get("iter_count", -1)),
        return_status=str(stats.get("return_status", "unknown")),
    )


def linearize_trajectory(
    states: np.ndarray,
    forces: np.ndarray,
    *,
    epsilon: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference the nonlinear discrete dynamics along a trajectory."""

    states = np.asarray(states, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64).reshape(-1)
    horizon = forces.size
    if states.shape != (horizon + 1, 4):
        raise ValueError("states must have shape (horizon + 1, 4)")
    matrices_a = np.empty((horizon, 4, 4), dtype=np.float64)
    matrices_b = np.empty((horizon, 4, 1), dtype=np.float64)
    for step in range(horizon):
        state = states[step : step + 1]
        force = np.array([forces[step]])
        for column in range(4):
            positive = state.copy()
            negative = state.copy()
            positive[:, column] += epsilon
            negative[:, column] -= epsilon
            matrices_a[step, :, column] = (
                nonlinear_single_cartpole_step(positive, force)
                - nonlinear_single_cartpole_step(negative, force)
            )[0] / (2.0 * epsilon)
        matrices_b[step, :, 0] = (
            nonlinear_single_cartpole_step(state, force + epsilon)
            - nonlinear_single_cartpole_step(state, force - epsilon)
        )[0] / (2.0 * epsilon)
    return matrices_a, matrices_b


def tvlqr_gains(
    states: np.ndarray,
    forces: np.ndarray,
    *,
    state_cost: np.ndarray | None = None,
    force_cost: float = 0.05,
    terminal_cost: np.ndarray | None = None,
) -> np.ndarray:
    """Run the finite-horizon Riccati recursion along a nonlinear trajectory."""

    if force_cost <= 0.0:
        raise ValueError("force_cost must be positive")
    matrices_a, matrices_b = linearize_trajectory(states, forces)
    cost_q = (
        np.diag([2.0, 80.0, 1.0, 8.0])
        if state_cost is None
        else np.asarray(state_cost, dtype=np.float64)
    )
    value = (
        cost_q * 20.0
        if terminal_cost is None
        else np.asarray(terminal_cost, dtype=np.float64).copy()
    )
    gains = np.empty((forces.size, 1, 4), dtype=np.float64)
    for step in range(forces.size - 1, -1, -1):
        matrix_a = matrices_a[step]
        matrix_b = matrices_b[step]
        denominator = force_cost + float((matrix_b.T @ value @ matrix_b).item())
        gain = (matrix_b.T @ value @ matrix_a) / denominator
        gains[step] = gain
        value = cost_q + matrix_a.T @ value @ (matrix_a - matrix_b @ gain)
    return gains


class TrajectoryReplayController:
    """Replay an optimized force sequence without feedback."""

    def __init__(self, forces: np.ndarray) -> None:
        self.forces = np.asarray(forces, dtype=np.float64).reshape(-1)

    def reset(self, num_envs: int) -> None:
        del num_envs
        self.step_index = 0

    def act(self, state: np.ndarray) -> np.ndarray:
        if self.step_index < self.forces.size:
            force = self.forces[self.step_index]
        else:
            force = 0.0
        self.step_index += 1
        return np.full(state.shape[0], force / ACTION_FORCE_SCALE_N)


class TVLQRTrackingController:
    """Track a nonlinear reference with time-varying Riccati feedback."""

    def __init__(
        self,
        states: np.ndarray,
        forces: np.ndarray,
        gains: np.ndarray,
        terminal_gain: np.ndarray,
    ) -> None:
        self.states = np.asarray(states, dtype=np.float64)
        self.forces = np.asarray(forces, dtype=np.float64).reshape(-1)
        self.gains = np.asarray(gains, dtype=np.float64)
        self.terminal_gain = np.asarray(terminal_gain, dtype=np.float64)

    def reset(self, num_envs: int) -> None:
        del num_envs
        self.step_index = 0

    def act(self, state: np.ndarray) -> np.ndarray:
        if self.step_index < self.forces.size:
            reference_state = self.states[self.step_index]
            reference_force = self.forces[self.step_index]
            gain = self.gains[self.step_index]
        else:
            reference_state = self.states[-1]
            reference_force = 0.0
            gain = self.terminal_gain
        error = np.asarray(state, dtype=np.float64) - reference_state
        error[:, 1] = np.arctan2(np.sin(error[:, 1]), np.cos(error[:, 1]))
        force = reference_force - (error @ gain.T)[:, 0]
        self.step_index += 1
        return np.clip(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)
