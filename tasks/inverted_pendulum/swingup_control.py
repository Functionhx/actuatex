"""Energy-based swing-up controllers for the single cart-pole.

The upright balance benchmark starts near ``theta = 0``.  This module defines
the separate nonlinear problem that starts near the hanging equilibrium
``theta = pi`` and must first inject energy before a local stabilizer can work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import (
    ACTION_FORCE_SCALE_N,
    CART_MASS_KG,
    POLE_LENGTH_M,
    POLE_MASS_KG,
    POLE_WIDTH_M,
)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap an angle without changing its continuous simulation coordinate."""

    angle = np.asarray(angle, dtype=np.float64)
    return np.arctan2(np.sin(angle), np.cos(angle))


def single_pole_energy(theta: np.ndarray, angular_velocity: np.ndarray) -> np.ndarray:
    """Return pole mechanical energy with zero potential at pivot height.

    ``theta = 0`` is upright, so the target energy is the maximum potential
    energy ``m g l_c``.  The inertia includes both the box inertia about its
    center and the parallel-axis term to the hinge.
    """

    center_length = POLE_LENGTH_M / 2.0
    center_inertia = POLE_MASS_KG * (POLE_WIDTH_M**2 + POLE_LENGTH_M**2) / 12.0
    hinge_inertia = center_inertia + POLE_MASS_KG * center_length**2
    kinetic = 0.5 * hinge_inertia * np.square(angular_velocity)
    potential = POLE_MASS_KG * 9.81 * center_length * np.cos(theta)
    return kinetic + potential


def upright_target_energy() -> float:
    """Mechanical energy of the stationary upright pole."""

    return POLE_MASS_KG * 9.81 * (POLE_LENGTH_M / 2.0)


@dataclass
class EnergySwingupController:
    """Pump pendulum energy while optionally shaping cart position.

    The continuous energy law follows from
    ``E_dot = -m*l*cos(theta)*theta_dot*x_ddot``.  Choosing acceleration
    proportional to ``(E-E_target)*cos(theta)*theta_dot`` drives the energy
    error toward zero.  Cart terms keep the finite rail usable, and a small
    deterministic kick escapes the exact hanging equilibrium.
    """

    energy_gain: float = 10.0
    cart_kp: float = 0.0
    cart_kd: float = 0.0
    target_energy_offset_j: float = 0.06
    kick_force_n: float = 1.5
    kick_speed_rad_s: float = 0.08

    def reset(self, num_envs: int) -> None:
        self.kick_direction = np.where(np.arange(num_envs) % 2, -1.0, 1.0)

    def force(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 4:
            raise ValueError("energy swing-up requires [x, theta, xdot, omega]")
        cart_position, theta, cart_velocity, angular_velocity = state.T
        energy_error = (
            single_pole_energy(theta, angular_velocity)
            - upright_target_energy()
            - self.target_energy_offset_j
        )
        acceleration_command = (
            self.energy_gain * energy_error * np.cos(theta) * angular_velocity
            - self.cart_kp * cart_position
            - self.cart_kd * cart_velocity
        )
        force = (CART_MASS_KG + POLE_MASS_KG) * acceleration_command

        near_hanging_rest = (np.abs(wrap_to_pi(theta)) > 0.85 * np.pi) & (
            np.abs(angular_velocity) < self.kick_speed_rad_s
        )
        force += near_hanging_rest * self.kick_force_n * self.kick_direction
        return np.clip(force, -ACTION_FORCE_SCALE_N, ACTION_FORCE_SCALE_N)

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.force(state) / ACTION_FORCE_SCALE_N


class HybridEnergyLQRController:
    """Energy swing-up plus hysteretic LQR capture around upright."""

    def __init__(
        self,
        balance_gain: np.ndarray,
        *,
        swingup: EnergySwingupController | None = None,
        capture_angle_rad: float = 0.30,
        capture_speed_rad_s: float = 2.5,
        release_angle_rad: float = 0.55,
        release_speed_rad_s: float = 4.0,
        capture_cart_position_m: float = 1.8,
        release_cart_position_m: float = 2.05,
    ) -> None:
        gain = np.asarray(balance_gain, dtype=np.float64)
        if gain.shape != (1, 4):
            raise ValueError(
                f"single-pole LQR gain must have shape (1, 4), got {gain.shape}"
            )
        if capture_angle_rad >= release_angle_rad:
            raise ValueError("capture angle must be smaller than release angle")
        self.balance_gain = gain
        self.swingup = swingup or EnergySwingupController(cart_kp=1.0, cart_kd=1.5)
        self.capture_angle_rad = capture_angle_rad
        self.capture_speed_rad_s = capture_speed_rad_s
        self.release_angle_rad = release_angle_rad
        self.release_speed_rad_s = release_speed_rad_s
        self.capture_cart_position_m = capture_cart_position_m
        self.release_cart_position_m = release_cart_position_m

    def reset(self, num_envs: int) -> None:
        self.balance_mode = np.zeros(num_envs, dtype=bool)
        self.swingup.reset(num_envs)

    def act(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        theta = wrap_to_pi(state[:, 1])
        angular_velocity = state[:, 3]
        cart_position = state[:, 0]
        enter = (
            (np.abs(theta) < self.capture_angle_rad)
            & (np.abs(angular_velocity) < self.capture_speed_rad_s)
            & (np.abs(cart_position) < self.capture_cart_position_m)
        )
        leave = (
            (np.abs(theta) > self.release_angle_rad)
            | (np.abs(angular_velocity) > self.release_speed_rad_s)
            | (np.abs(cart_position) > self.release_cart_position_m)
        )
        self.balance_mode = (self.balance_mode & ~leave) | enter

        swingup_force = self.swingup.force(state)
        balance_state = state.copy()
        balance_state[:, 1] = theta
        balance_force = -(balance_state @ self.balance_gain.T)[:, 0]
        force = np.where(self.balance_mode, balance_force, swingup_force)
        return np.clip(force / ACTION_FORCE_SCALE_N, -1.0, 1.0)
