"""Test standing stability under zero action with the actuated model.

Tries the position-actuator + damping + armature fix at a few armature values.
Target: base z stable at ~0.20-0.24 m for >=5 s with zero action.
"""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_builder import build_floating_model, JOINT_NAMES
from observation_builder import DEFAULT_ANGLES, SIM_DT, DECIMATION, INIT_POS
from actuatex_paths import ROBOT_URDF

URDF = str(ROBOT_URDF)


def test_stand(armature=0.01, kv=0.5, integrator="implicitfast",
               solref_timeconst=None, iterations=None, duration=5.0):
    model = build_floating_model(URDF, armature=armature, kv=kv,
                                 integrator=integrator,
                                 solref_timeconst=solref_timeconst,
                                 iterations=iterations)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    # Map joints to qpos/dof addresses.
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qposadr = np.array([model.jnt_qposadr[i] for i in joint_ids])
    dofadr = np.array([model.jnt_dofadr[i] for i in joint_ids])
    default_q = np.array([DEFAULT_ANGLES[n] for n in JOINT_NAMES])

    # Position actuators: ctrl[i] is the target for actuator i (declaration order
    # = JOINT_NAMES order). Verify actuator order matches joint order.
    for i, jn in enumerate(JOINT_NAMES):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jn}_pos")
        assert aid == i, f"actuator {jn}_pos at {aid}, expected {i}"

    # Spawn.
    data.qpos[:3] = INIT_POS
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[qposadr] = default_q
    data.qvel[:] = 0.0
    data.ctrl[:] = default_q  # zero action -> ctrl = default
    mujoco.mj_forward(model, data)

    z0 = data.qpos[2]
    zs = []
    n_control = int(duration / (SIM_DT * DECIMATION))
    for k in range(n_control):
        data.ctrl[:] = default_q  # hold zero-action target
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        zs.append(data.qpos[2])

    zs = np.array(zs)
    z_final = zs[-1]
    z_mean_last2s = np.mean(zs[int(-2.0/(SIM_DT*DECIMATION)):]) if len(zs) > 100 else np.mean(zs)
    z_std_last2s = np.std(zs[int(-2.0/(SIM_DT*DECIMATION)):]) if len(zs) > 100 else np.std(zs)
    return z0, z_final, z_mean_last2s, z_std_last2s


if __name__ == "__main__":
    print(f"{'config':<55} {'z0':>6} {'z_end':>6} {'z_2s':>6} {'std_2s':>7}")
    configs = [
        dict(armature=0.01, kv=0.5, integrator="implicitfast"),
        dict(armature=0.05, kv=0.5, integrator="implicitfast"),
        dict(armature=0.1,  kv=0.5, integrator="implicitfast"),
        dict(armature=0.01, kv=0.5, integrator="Euler"),
        dict(armature=0.01, kv=0.5, integrator="implicitfast", solref_timeconst=0.01),
        dict(armature=0.01, kv=0.5, integrator="implicitfast", iterations=50),
        dict(armature=0.1,  kv=2.0, integrator="implicitfast"),
    ]
    for cfg in configs:
        try:
            z0, zf, zm, zs = test_stand(**cfg)
            tag = ",".join(f"{k}={v}" for k, v in cfg.items())
            print(f"{tag:<55} {z0:6.3f} {zf:6.3f} {zm:6.3f} {zs:7.4f}")
        except Exception as e:
            tag = ",".join(f"{k}={v}" for k, v in cfg.items())
            print(f"{tag:<55} ERROR: {e}")
