"""Backend-agnostic observation builder + constants for TinyMal sim2sim.

Pure numpy, no Isaac Gym import, so it runs under any Python (unitree-rl for
MuJoCo, Isaac Sim's bundled Python). All values verified against training code:
- obs composition: legged_robot.py:182-196 (compute_observations)
- obs scales / clip: legged_robot_config.py:127-135
- commands_scale: legged_robot.py:458 ([lin_vel, lin_vel, ang_vel])
- default joint angles: tinymal_config.py:14-27
- DOF order = URDF joint declaration order: FL,FR,RL,RR x hip,thigh,calf
"""

import numpy as np

# Policy DOF order = URDF joint declaration order (verified from tinymal.urdf).
POLICY_DOF_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

DEFAULT_ANGLES = {
    "FL_hip_joint": -0.16, "FR_hip_joint": 0.16,
    "RL_hip_joint": -0.16, "RR_hip_joint": 0.16,
    "FL_thigh_joint": 0.68, "FR_thigh_joint": 0.68,
    "RL_thigh_joint": 0.68, "RR_thigh_joint": 0.68,
    "FL_calf_joint": 1.3, "FR_calf_joint": 1.3,
    "RL_calf_joint": 1.3, "RR_calf_joint": 1.3,
}
DEFAULT_DOF_POS = np.array(
    [DEFAULT_ANGLES[n] for n in POLICY_DOF_ORDER], dtype=np.float64
)

# Observation scales (legged_robot_config.py normalization.obs_scales).
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
COMMANDS_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float64)  # vx, vy, yaw
CLIP_OBS = 100.0

# Control (tinymal_config.py control class).
KP = 20.0
KD = 0.5
ACTION_SCALE = 0.25
TORQUE_LIMIT = 12.0  # N*m, from URDF effort
DECIMATION = 4
SIM_DT = 0.005
POLICY_DT = SIM_DT * DECIMATION  # 0.02 s = 50 Hz control

INIT_POS = np.array([0.0, 0.0, 0.28])  # base spawn (tinymal_config init_state)
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0])  # unit gravity (z-up)


def quat_rotate_inverse(q_xyzw, v):
    """World->body rotation of vector v by quaternion q (xyzw layout).

    Mirrors isaacgym.torch_utils.quat_rotate_inverse: v_body = q^{-1} (x) v.
    With n=q.xyz, w=q.w, t=2*cross(n,v): result = v - w*t + cross(n,t).
    """
    n = np.array([q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    w = q_xyzw[3]
    t = 2.0 * np.cross(n, v)
    return v - w * t + np.cross(n, t)


def project_gravity(q_xyzw):
    """Gravity vector expressed in the body frame (obs terms 6:9)."""
    return quat_rotate_inverse(q_xyzw, GRAVITY_WORLD)


def build_obs(base_lin_vel, base_ang_vel, projected_gravity, commands,
              dof_pos, dof_vel, last_action):
    """Assemble the 48-dim observation exactly as in training.

    base_lin_vel / base_ang_vel / projected_gravity are body-frame (3,).
    commands is (>=3,) [vx, vy, yaw_rate]. dof_pos/dof_vel/last_action are (12,)
    in POLICY_DOF_ORDER.
    """
    obs = np.concatenate([
        base_lin_vel * OBS_SCALE_LIN_VEL,                     # 0:3
        base_ang_vel * OBS_SCALE_ANG_VEL,                     # 3:6
        projected_gravity,                                    # 6:9
        np.asarray(commands)[:3] * COMMANDS_SCALE,            # 9:12
        (dof_pos - DEFAULT_DOF_POS) * OBS_SCALE_DOF_POS,      # 12:24
        dof_vel * OBS_SCALE_DOF_VEL,                          # 24:36
        last_action,                                          # 36:48
    ]).astype(np.float64)
    return np.clip(obs, -CLIP_OBS, CLIP_OBS)


def dof_permutation(sim_joint_names):
    """Map a backend's joint-name order -> policy DOF order.

    Returns an index array `perm` such that `sim_dof[perm]` is in
    POLICY_DOF_ORDER. Raises if any expected joint is missing.
    """
    sim = list(sim_joint_names)
    perm = []
    for name in POLICY_DOF_ORDER:
        if name not in sim:
            raise ValueError(f"Joint {name!r} not in backend joints: {sim}")
        perm.append(sim.index(name))
    return np.array(perm, dtype=np.int64)
