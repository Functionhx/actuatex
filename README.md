<div align="center">

# ActuateX

### Learn. Act. Control.

**An experimental reinforcement-learning platform for robust control across simulators, robots, and real-world systems.**

</div>

ActuateX trains and evaluates the same TinyMal quadruped across Isaac Gym, Isaac Lab on Isaac Sim, and MuJoCo. The current research target is simple: make the dog walk, climb, resist disturbances, and transfer between physics engines without hiding failures behind a single training reward.

> Status: active research prototype. Simulation only; not safety-certified for hardware deployment.

## What is included

| Backend | Role | ActuateX additions |
|---|---|---|
| Isaac Gym | high-throughput PhysX baseline | TinyMal task family, robust/stair curricula, push recovery, distillation, off-screen recording |
| Isaac Lab + Isaac Sim | current NVIDIA robotics stack | native Manager-Based environments, observation/control alignment, robust and stair tasks, offline compatibility |
| MuJoCo | independent dynamics engine | native PPO training, actuated model builder, stair/push tests, forward and reverse sim2sim |

Isaac Sim and Isaac Gym are not the same product. Isaac Sim is the full Omniverse robotics simulator; Isaac Lab is the current learning framework built on it. Isaac Gym Preview is the older standalone GPU simulator. ActuateX keeps separate implementations because a policy that succeeds in one physics stack is not automatically robust in another.

## Current TinyMal evidence

![Native capabilities](docs/media/final_native_capabilities.png)

| Test | Result |
|---|---:|
| Isaac Gym → MuJoCo, six command segments | mean RMSE 0.08024; 0/6 falls |
| Isaac Lab → MuJoCo, six command segments | mean RMSE 0.33272; 2/6 falls |
| MuJoCo native policy | mean RMSE 0.07957; 0/6 falls |
| MuJoCo dynamics grid | 16/16 no-fall cases |
| MuJoCo sustained pushes | 11/16 accepted cases |
| Isaac Lab native stair showcase | 64/64 successes under showcase settings |

The stricter Isaac Lab transfer result is intentionally retained. It exposes the remaining solver, actuator, and system-identification gap instead of presenting only the best-looking run.

## Repository map

```text
backends/isaac_gym/   TinyMal overlay plus auditable upstream patches
backends/isaac_lab/   native Isaac Lab task package and scripts
backends/mujoco/      standalone MuJoCo training and evaluation
robots/tinymal/       canonical URDF and meshes
tools/sim2sim/        policy and cross-engine evaluation tools
tools/checkpoints/    reproducible checkpoint selection
scripts/              guarded integration installers
docs/                 architecture, results, and code-change report
```

Large dependencies are deliberately absent. Isaac Sim and Isaac Gym binaries are installed under their own licenses; Isaac Lab, RSL-RL, and `unitree_rl_gym` are pinned upstream and patched locally. Checkpoints, logs, videos, and generated USD/MJCF files belong in `artifacts/` or a GitHub Release, not Git history.

## Quick start

Clone the pinned open-source dependencies:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym.git _deps/unitree_rl_gym
git -C _deps/unitree_rl_gym checkout 276801e46c5d433564f24658bac64f254b7d2d4b

git clone https://github.com/leggedrobotics/rsl_rl.git _deps/rsl_rl
git -C _deps/rsl_rl checkout v1.0.2
```

Apply the Isaac Gym integration after installing NVIDIA Isaac Gym Preview 4:

```bash
python scripts/install_isaac_gym_overlay.py \
  --unitree-root _deps/unitree_rl_gym \
  --rsl-rl-root _deps/rsl_rl
pip install -e _deps/rsl_rl -e _deps/unitree_rl_gym
python _deps/unitree_rl_gym/legged_gym/scripts/train.py \
  --task=tinymal --headless
```

For MuJoCo:

```bash
pip install -r backends/mujoco/requirements.txt
pip install -e _deps/rsl_rl
RSL_RL_ROOT=_deps/rsl_rl \
python backends/mujoco/train_mujoco.py --num_envs 64 --max_iters 1500
```

Isaac Lab requires the pinned checkout and a compatible Isaac Sim installation. See the backend-specific guides:

- [Isaac Gym setup](backends/isaac_gym/README.md)
- [Isaac Lab / Isaac Sim setup](backends/isaac_lab/README.md)
- [MuJoCo setup](backends/mujoco/README.md)
- [Chinese code modification report](docs/CODE_CHANGES_REPORT.zh-CN.md)
- [PDF code modification report](docs/ActuateX_代码修改报告.pdf)

## Research direction

Near-term work focuses on stronger system identification, recurrent or history-based policies, symmetry regularization, teacher-student adaptation, larger domain randomization, and one acceptance suite shared by all simulators. Hardware deployment will only follow actuator, latency, emergency-stop, and fall-safety validation.

## Licensing and provenance

ActuateX does not redistribute NVIDIA simulator binaries. Patches derived from Isaac Lab, Unitree RL Gym, and RSL-RL retain their upstream BSD-3-Clause notices under `third_party/licenses/`. The TinyMal asset came from the supplied course materials and currently has no explicit redistribution license; review `robots/tinymal/ASSET_NOTICE.md` before making a public release. No project-wide open-source license has been declared yet.
