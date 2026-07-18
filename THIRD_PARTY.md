# Third-party software and assets

ActuateX uses, but does not vendor, the following major dependencies:

| Dependency | Upstream | License / distribution |
|---|---|---|
| Isaac Gym Preview 4 | NVIDIA developer download | Separate NVIDIA license; binaries are not included |
| Isaac Sim | NVIDIA Omniverse / Isaac Sim | Separate NVIDIA license; binaries are not included |
| Isaac Lab | <https://github.com/isaac-sim/IsaacLab> | BSD-3-Clause; pinned in `backends/isaac_lab/upstream.json` |
| Unitree RL Gym | <https://github.com/unitreerobotics/unitree_rl_gym> | BSD-3-Clause; pinned in `backends/isaac_gym/upstream.json` |
| RSL-RL | <https://github.com/leggedrobotics/rsl_rl> | BSD-3-Clause; v1.0.2 pinned in `backends/isaac_gym/upstream.json` |
| MuJoCo | <https://github.com/google-deepmind/mujoco> | Apache-2.0; installed as a Python dependency |

Copies of the BSD license texts required by the included compatibility patches are in `third_party/licenses/`.

The TinyMal URDF and meshes were supplied as course assets. Their files contain no complete redistribution grant. They are included in this initial private repository for reproducibility; confirm ownership and licensing or replace them before a public release.
