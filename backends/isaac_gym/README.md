# Isaac Gym backend

This backend is an overlay for the pinned `unitree_rl_gym` and RSL-RL v1.0.2 sources. NVIDIA Isaac Gym Preview 4 must be installed separately.

## Prepare dependencies

```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym.git _deps/unitree_rl_gym
git -C _deps/unitree_rl_gym checkout 276801e46c5d433564f24658bac64f254b7d2d4b

git clone https://github.com/leggedrobotics/rsl_rl.git _deps/rsl_rl
git -C _deps/rsl_rl checkout v1.0.2

python scripts/install_isaac_gym_overlay.py \
  --unitree-root _deps/unitree_rl_gym \
  --rsl-rl-root _deps/rsl_rl

pip install -e _deps/rsl_rl -e _deps/unitree_rl_gym
```

The installer dry-runs every patch, recognizes already-applied patches, copies the TinyMal task files, and installs the robot asset into the locations expected by Unitree RL Gym. It refuses unknown source layouts instead of modifying an arbitrary directory.

## Tasks

| Task | Purpose |
|---|---|
| `tinymal` | flat-ground velocity baseline |
| `tinymal_push` | targeted disturbance recovery |
| `tinymal_stairs` | stair curriculum |
| `tinymal_robust` | flat-ground domain randomization and pushes |
| `tinymal_robust_stairs` | robust stair training |
| `tinymal_robust_mixed` | mixed flat/stair robust training |

Baseline training:

```bash
python _deps/unitree_rl_gym/legged_gym/scripts/train.py \
  --task=tinymal --headless --num_envs=4096
```

Robust training with safe checkpoint resume:

```bash
python _deps/unitree_rl_gym/legged_gym/scripts/train_tinymal_robust.py \
  --task=tinymal_robust --headless --num_envs=4096

TINYMAL_RESUME_CHECKPOINT=artifacts/checkpoints/isaac_gym/model_500.pt \
python _deps/unitree_rl_gym/legged_gym/scripts/train_tinymal_robust.py \
  --task=tinymal_robust --headless
```

Use the scripts copied into `_deps/unitree_rl_gym/legged_gym/scripts/` for stair transfer, push recovery, PD ablation, stress testing, and evaluation.

## Upstream changes

- `0001-register-tinymal-tasks.patch` adds six registry entries.
- `0002-offscreen-rendering.patch` allows headless camera rendering without an interactive viewer.
- `0003-reference-policy-distillation.patch` adds two optional, task-selective teacher losses to PPO.

All patches are disabled or behavior-preserving unless their corresponding ActuateX feature is requested.
