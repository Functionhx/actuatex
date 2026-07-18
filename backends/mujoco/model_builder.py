"""Build the floating-base TinyMal model with proper MuJoCo actuation.

The raw URDF import gives 12 hinge joints with zero armature, zero damping,
and no actuators (nu=0). Applying PD via qfrc_applied on these low-inertia
joints is numerically unstable (explicit Euler), so the robot collapses from
0.28 m to ~0.07 m even at zero action.

Fix: add <position> actuators (kp=20, forcelimit=12) + joint damping=0.5
(the D term, integrated implicitly by MuJoCo) + small armature, and switch
to the implicitfast integrator. This matches Isaac Gym's PD exactly:
    tau = kp * (q_target - q) - kd * dqvel, clipped to +-12 N*m

Functions exported:
    build_floating_model(urdf_path, **tuning) -> MjModel
    STAND configuration constants are in observation_builder.py.
"""

import os
import tempfile
import numpy as np
import mujoco


def _load_compiled_mjcf(urdf_path, save_path=None):
    """Compile the URDF (fixing meshdir) and restructure to add floating base."""
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
    if save_path:
        open(save_path, "w").write(new_s)
    return new_s


# Hinge joint names in MuJoCo order (matches URDF declaration = policy order).
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


def build_floating_model(urdf_path,
                         kp=20.0, kv=0.5, forcelimit=12.0,
                         armature=0.01, joint_damping=None,
                         integrator="implicitfast",
                         solref_timeconst=None,
                         foot_friction=None,
                         iterations=None,
                         worldbody_extras=None,
                         save_path=None):
    """Load URDF, add floating base, position actuators, joint damping/armature.

    Args:
        kp: position servo gain (matches Isaac Gym Kp).
        kv: velocity damping — added as joint `damping` so MuJoCo integrates it
            implicitly (matches Isaac Gym Kd).
        forcelimit: actuator force limit (N*m).
        armature: rotor inertia reflected to each joint (stabilizes low-inertia
            joints under PD). 0.01 is a common small-quad default.
        joint_damping: if not None, overrides kv for joint damping.
        integrator: "implicitfast" (default, stable for PD) or "Euler".
        solref_timeconst: if set, overrides contact solref time constant.
        foot_friction: if set, overrides foot geom friction (3-tuple).
        iterations: if set, overrides solver iterations.
        worldbody_extras: optional MJCF XML inserted as additional static world
            bodies/geoms (used by cross-engine stair evaluation).
        save_path: optional path for the generated, fully actuated MJCF.
    """
    s = _load_compiled_mjcf(urdf_path)

    if worldbody_extras:
        worldbody_end = s.rfind("</worldbody>")
        if worldbody_end < 0:
            raise ValueError("compiled MJCF has no closing worldbody tag")
        s = s[:worldbody_end] + worldbody_extras + "\n  " + s[worldbody_end:]

    # Inject <default> for joint damping/armature and switch integrator.
    damp = kv if joint_damping is None else joint_damping
    # Build a <default> block that applies damping+armature to all hinge joints.
    # We insert it right after <compiler> in the compiled MJCF.
    default_block = (
        f'<default>\n'
        f'      <joint damping="{damp}" armature="{armature}"/>\n'
        f'      <geom rgba="0.7 0.7 0.7 1"/>\n'
        f'    </default>\n  '
    )
    # Insert before the first <worldbody> — actually defaults go in <mujoco>,
    # right after compiler/options. Insert after the <compiler .../> line.
    compiler_end = s.find("/>")  # end of <compiler .../>
    # find the compiler tag end properly
    cidx = s.find("<compiler")
    cend = s.find("/>", cidx) + 2
    s = s[:cend] + "\n  " + default_block + s[cend:]

    # Switch integrator in <option> (or add one).
    opt_start = s.find("<option")
    if opt_start >= 0:
        opt_end = s.find("/>", opt_start) + 2
        opt_line = s[opt_start:opt_end]
        if "integrator" in opt_line:
            import re
            opt_line = re.sub(r'integrator="[^"]*"',
                              f'integrator="{integrator}"', opt_line)
        else:
            opt_line = opt_line[:-2] + f' integrator="{integrator}"/>'
        s = s[:opt_start] + opt_line + s[opt_end:]
    else:
        # add <option> after the compiler
        cidx2 = s.find("<compiler")
        cend2 = s.find("/>", cidx2) + 2
        s = s[:cend2] + f'\n  <option integrator="{integrator}"/>' + s[cend2:]

    # Optionally bump solver iterations.
    if iterations is not None:
        s = s.replace("<option", f'<option iterations="{iterations}"', 1)

    # Add <actuator> block before </mujoco>.
    actuator_lines = []
    for jn in JOINT_NAMES:
        actuator_lines.append(
            f'    <position name="{jn}_pos" joint="{jn}" kp="{kp}" '
            f'forcerange="-{forcelimit} {forcelimit}"/>'
        )
    actuator_block = "\n  <actuator>\n" + "\n".join(actuator_lines) + "\n  </actuator>\n"
    s = s.replace("</mujoco>", actuator_block + "</mujoco>")

    # Optionally tune contact solref on all geoms (insert into <default><geom>).
    if solref_timeconst is not None:
        s = s.replace('<geom rgba="0.7 0.7 0.7 1"/>',
                      f'<geom solref="{solref_timeconst} 1" rgba="0.7 0.7 0.7 1"/>', 1)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as stream:
            stream.write(s)
    return mujoco.MjModel.from_xml_string(s)
