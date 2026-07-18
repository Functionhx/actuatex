"""Inspect the floating-base TinyMal model in MuJoCo and reproduce the
standing-collapse catch (zero action, raw qfrc_applied PD).

Saves the compiled MJCF so we can see what MuJoCo made of the URDF
(actuators, joints, armature, contact).
"""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from observation_builder import (
    DEFAULT_DOF_POS, DEFAULT_ANGLES, KP, KD, ACTION_SCALE,
    TORQUE_LIMIT, DECIMATION, SIM_DT, INIT_POS,
)
from actuatex_paths import ARTIFACTS_ROOT, ROBOT_URDF

URDF = str(ROBOT_URDF)


def build_floating_model(urdf_path, save_path=None):
    import tempfile
    txt = open(urdf_path, "r", encoding="utf-8").read()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    meshes_dir = os.path.join(os.path.dirname(urdf_dir), "meshes")
    txt = txt.replace('meshdir="./meshes"', f'meshdir="{meshes_dir}"')
    compiled = mujoco.MjModel.from_xml_string(txt)
    tmp = tempfile.mktemp(suffix=".mjcf")
    mujoco.mj_saveLastXML(tmp, compiled)
    s = open(tmp, "r", encoding="utf-8").read()
    a = s.find("<worldbody>") + len("<worldbody>")
    b = s.rfind("</worldbody>")
    inner = s[a:b].strip()
    base_inertial = (
        '<inertial pos="0.0034198 6.4226e-06 0.0033633" mass="2.2657" '
        'fullinertia="0.0011588 0.0028416 0.0032559 4.4374e-07 -6.9655e-07 -9.2423e-07"/>'
    )
    floor = ('<geom name="floor" type="plane" size="0 0 0.1" '
             'friction="1 0.005 0.0001" rgba="0.4 0.4 0.4 1"/>')
    body = (f'<body name="base" pos="0 0 0"><freejoint name="root"/>{base_inertial}\n'
            f'      {inner}\n    </body>')
    new_s = s[:a] + "\n      " + floor + "\n      " + body + "\n    " + s[b:]
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(new_s, encoding="utf-8")
    return mujoco.MjModel.from_xml_string(new_s)


def main():
    model = build_floating_model(
        URDF, ARTIFACTS_ROOT / "mujoco" / "compiled_raw.mjcf"
    )
    model.opt.timestep = SIM_DT
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} njnt={model.njnt}")
    print(f"na (actuated dof)={model.na}")
    print(f"opt.timestep={model.opt.timestep}")
    print(f"=== joints ===")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  jnt {i}: {name} type={model.jnt_type[i]} "
              f"qposadr={model.jnt_qposadr[i]} dofadr={model.jnt_dofadr[i]} "
              f"axis={model.jnt_axis[i]}")
    print(f"=== actuators ({model.nu}) ===")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  act {i}: {name} trntype={model.actuator_trntype[i]} "
              f"gear={model.actuator_gear[i]} ctrlrange={model.actuator_ctrlrange[i]} "
              f"forcelimit={model.actuator_forcelimit[i]}")
    print(f"=== dof armature/damping/friction ===")
    for i in range(model.nv):
        print(f"  dof {i}: armature={model.dof_armature[i]} "
              f"damping={model.dof_damping[i]} frictionloss={model.dof_frictionloss[i]}")

    hinge_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in hinge_ids]
    qposadr = np.array([model.jnt_qposadr[i] for i in hinge_ids])
    dofadr = np.array([model.jnt_dofadr[i] for i in hinge_ids])
    print(f"hinge joint names (mujoco order): {joint_names}")
    default_mujoco = np.array([DEFAULT_ANGLES[n] for n in joint_names])

    data = mujoco.MjData(model)
    data.qpos[:3] = INIT_POS
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[qposadr] = default_mujoco
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"\n=== spawn: base z = {data.qpos[2]:.4f} ===")

    # zero-action PD for 5 s
    q_target = default_mujoco  # zero action
    n_control = int(5.0 / (SIM_DT * DECIMATION))
    for k in range(n_control):
        for _ in range(DECIMATION):
            q = data.qpos[qposadr]
            dq = data.qvel[dofadr]
            tau = KP * (q_target - q) - KD * dq
            tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
            data.qfrc_applied[dofadr] = tau
            mujoco.mj_step(model, data)
        if k % 25 == 0 or k == n_control - 1:
            tau_curr = data.qfrc_applied[dofadr]
            print(f"  t={k*SIM_DT*DECIMATION:.2f}s  z={data.qpos[2]:.4f}  "
                  f"max|tau|={np.max(np.abs(tau_curr)):.3f}  "
                  f"qpos_err_rms={np.sqrt(np.mean((data.qpos[qposadr]-default_mujoco)**2)):.4f}")
    print(f"\n=== after 5s zero-action: base z = {data.qpos[2]:.4f} (target ~0.24) ===")


if __name__ == "__main__":
    main()
