# Isaac Lab / Isaac Sim backend

Isaac Sim is the simulator and application runtime. Isaac Lab is the reinforcement-learning framework built on top of it. This backend contains the native TinyMal Isaac Lab task; it does not redistribute either product.

## Tested stack

- Isaac Lab commit `b4c321024792976150ca55fddb26fa34480d974e`
- Isaac Sim `5.1.0-rc.19`
- RSL-RL through the compatible Isaac Lab installation

Follow the official Isaac Lab installation instructions for the matching Isaac Sim build, then prepare the pinned checkout:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git _deps/IsaacLab
git -C _deps/IsaacLab checkout b4c321024792976150ca55fddb26fa34480d974e

python scripts/install_isaac_lab_compat.py \
  --isaac-lab-root _deps/IsaacLab
```

The compatibility patch guards an importer API missing in the tested Isaac Sim release and supplies an offline cuboid ground when the default Nucleus grid asset is unavailable.

## Train

Set `ISAAC_SIM_PYTHON` to the Python launcher belonging to the installed Isaac Sim, and ensure the pinned Isaac Lab packages are installed in that environment.

```bash
export ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh

"$ISAAC_SIM_PYTHON" backends/isaac_lab/scripts/train_tinymal.py \
  --task Isaac-Velocity-Native-Omni-TinyMal-v0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --headless --seed 1
```

Training logs default to `artifacts/isaac_lab/logs/rsl_rl/`. Set `ACTUATEX_ARTIFACTS` to place them on another disk.

## Evaluate and record stairs

```bash
"$ISAAC_SIM_PYTHON" backends/isaac_lab/scripts/evaluate_tinymal_stairs.py \
  --ckpt artifacts/checkpoints/isaac_lab/model.pt \
  --num_envs 64 --video --headless \
  --video_dir artifacts/videos/isaac_lab_stairs
```

Add `--robust` to enable the stricter randomized evaluation. The showcase and robust conditions should be reported separately.

## Transfer an Isaac Gym policy

```bash
"$ISAAC_SIM_PYTHON" backends/isaac_lab/scripts/play_old_policy.py \
  --ckpt artifacts/checkpoints/isaac_gym/model.pt \
  --suite --num_envs 64 --headless \
  --out artifacts/isaac_lab/gym_transfer.json
```

The 48-dimensional observation layout, joint order, action scale, torque clipping, and control period are explicitly aligned with the Isaac Gym backend.
