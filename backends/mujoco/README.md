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

## Inverted-pendulum control arena

The 1/2/3-link benchmark compares classical control, estimators, PPO, nonlinear swing-up, and robustness under one shared contract. Start with the formal balance and swing-up suites:

```bash
RSL_RL_ROOT=/path/to/rsl_rl \
python backends/mujoco/evaluate_classical_control_suite.py \
  --orders 1 2 3 --episodes 1024

python backends/mujoco/evaluate_swingup_suite.py --episodes 1024
python backends/mujoco/evaluate_trajectory_control_suite.py \
  --horizon 120 --episodes 1024
```

The paired robustness matrix includes mass, damping, actuator, delay, sensor-noise, push, and combined shifts, with the native PPO actor in the same table:

```bash
RSL_RL_ROOT=/path/to/rsl_rl \
python backends/mujoco/evaluate_robust_control_suite.py \
  --orders 1 2 3 --episodes 256 \
  --methods lqr h_infinity sliding_mode mujoco_ppo
```

Read the [full benchmark report](../../docs/INVERTED_PENDULUM_BENCHMARK.zh-CN.md) before comparing methods from different lanes. Record the optimized TVLQR swing-up in PPT-ready H.264 with:

```bash
MUJOCO_GL=egl \
python backends/mujoco/record_inverted_pendulum_swingup.py
```

## Record the PPT staircase video

`ffmpeg` and an EGL-capable MuJoCo setup are required:

```bash
MUJOCO_GL=egl python backends/mujoco/record_mujoco_stairs.py \
  --checkpoint artifacts/checkpoints/mujoco/model.pt \
  --output artifacts/videos/tinymal_mujoco_stairs.mp4 \
  --poster artifacts/videos/tinymal_mujoco_stairs.png
```

The recorder exits with failure if the rollout does not reach the top, stay within the centerline tolerance, and remain upright.

## Serial wheel-legged sim2sim

The Isaac Sim 6 wheel-legged actor has a separate, auditable 28-observation / 6-action contract and an explicit MJCF twin. Run the raw transfer first so command-reversal failures remain visible:

```bash
python backends/mujoco/wheel_legged_sim2sim.py \
  --out artifacts/mujoco/sim2sim/wheel_legged/raw.json \
  --tracking artifacts/mujoco/sim2sim/wheel_legged/raw.csv
```

The tested deployment wrapper limits command acceleration without changing policy weights:

```bash
python backends/mujoco/wheel_legged_sim2sim.py \
  --linear-command-slew 6 --yaw-command-slew 4 \
  --out artifacts/mujoco/sim2sim/wheel_legged/deployed.json \
  --tracking artifacts/mujoco/sim2sim/wheel_legged/deployed.csv

python backends/mujoco/sweep_wheel_legged_sim2sim.py
```

The canonical sweep covers raw transfer, the command-rate boundary, 5--20 ms delays, base-mass scaling, and friction. Read [`WHEEL_LEGGED_SIM2SIM.zh-CN.md`](../../docs/WHEEL_LEGGED_SIM2SIM.zh-CN.md) before interpreting the non-monotonic contact results.

Record the zero-fall 17.5-second showcase with:

```bash
MUJOCO_GL=egl python backends/mujoco/wheel_legged_sim2sim.py \
  --sequence showcase \
  --linear-command-slew 4 --yaw-command-slew 3 \
  --camera-distance 1.4 --camera-elevation -12 \
  --video artifacts/mujoco/videos/\
ActuateX_SerialWheelLegged_MuJoCo_Sim2Sim.mp4
```

Run the contract regression suite with global pytest plugins disabled; some ROS 2 distributions install an older `launch_testing` plugin globally:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -q backends/mujoco/tests/test_wheel_legged_contract.py
```
