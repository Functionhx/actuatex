# MuJoCo backend

This backend can train TinyMal natively in MuJoCo and can evaluate policies trained in either NVIDIA backend. It uses the same 48-dimensional observation, 12-dimensional action, 50 Hz policy rate, and nominal PD controller as the Isaac Gym baseline.

## Install

```bash
pip install -r backends/mujoco/requirements.txt
pip install -e _deps/rsl_rl
```

RSL-RL defaults to `_deps/rsl_rl`. Override paths without editing code:

```bash
export RSL_RL_ROOT=/path/to/rsl_rl
export ACTUATEX_ARTIFACTS=/fast/disk/actuatex-artifacts
export ACTUATEX_TINYMAL_URDF=/optional/custom/tinymal.urdf
```

## Smoke test and train

```bash
python backends/mujoco/test_stand.py
python backends/mujoco/bench_env.py

python backends/mujoco/train_mujoco.py \
  --num_envs 64 --max_iters 1500 \
  --learning_rate 3e-4 --command_mode omni
```

The canonical final checkpoint is written to `artifacts/checkpoints/mujoco/model.pt`; intermediate logs remain under `artifacts/mujoco/`.

## Evaluate

```bash
python backends/mujoco/eval_mujoco.py \
  --checkpoint artifacts/checkpoints/mujoco/model.pt \
  --out_dir artifacts/mujoco/evaluation

python backends/mujoco/eval_mujoco_tasks.py \
  --checkpoint artifacts/checkpoints/mujoco/model.pt \
  --out_dir artifacts/mujoco/tasks
```

The first command runs the shared six-segment command suite. The second tests strict stair ascent and a 4-direction × 4-magnitude sustained-force matrix.

## Record the PPT staircase video

`ffmpeg` and an EGL-capable MuJoCo setup are required:

```bash
MUJOCO_GL=egl python backends/mujoco/record_mujoco_stairs.py \
  --checkpoint artifacts/checkpoints/mujoco/model.pt \
  --output artifacts/videos/tinymal_mujoco_stairs.mp4 \
  --poster artifacts/videos/tinymal_mujoco_stairs.png
```

The recorder exits with failure if the rollout does not reach the top, stay within the centerline tolerance, and remain upright.
