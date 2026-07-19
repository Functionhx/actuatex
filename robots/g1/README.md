# Unitree G1 29-DOF

ActuateX uses the G1 29-DOF model as its first humanoid industrialization
target.  The repository does **not** redistribute the upstream meshes, USD or
MJCF.  Their exact source revisions and licenses are recorded in
[`upstream.json`](./upstream.json).

## Why 29 DOF

The 29-DOF layout has a documented Unitree SDK motor order and an official
MuJoCo twin.  That lets one policy contract cross four boundaries without
guessing:

1. Isaac Lab / PhysX training;
2. MuJoCo or MuJoCo Playground independent evaluation;
3. ONNX export and C++ replay;
4. Unitree SDK2 `LowCmd` / `LowState` deployment.

The canonical ActuateX order is always the hardware SDK order.  The current
Unitree RL Lab USD traversal uses a different interleaved order; the conversion
is explicit in [`g1_29dof.py`](../../tasks/locomotion/g1_29dof.py), never hidden
inside a checkpoint.

## Frozen policy contract

| Item | Contract |
|---|---:|
| Action | 29 normalized joint-position residuals |
| Control | PD target, 0.25 rad action scale |
| Physics / policy step | 0.005 s / 0.020 s (50 Hz) |
| One observation frame | 96 values |
| History | 5 samples, **term-major** layout |
| Actor input | 480 values |
| Motor model | Direction-aware torque-speed curve + friction |
| Runtime guard | command slew, tilt gate, joint limits, stale-state watchdog |

The observation does not expose simulator-only base linear velocity to the
actor.  The critic may receive privileged state during training, but exported
actors must run from IMU, joint state, command and previous-action history.

## Current evidence

- The official Isaac Lab G1 task has completed a local 4-environment,
  1-iteration PPO smoke run.  It proves the Isaac Sim/PhysX pipeline and asset
  download, not final walking quality.
- The backend-neutral 29-DOF contract has unit coverage for joint permutation,
  upstream default pose, 480-D history layout, exact action delay,
  torque-speed clipping and fail-closed safety behavior.
- The official `unitree_mujoco` G1 joint, actuator and sensor definitions have
  been inspected against the SDK order.
- Native 29-DOF training and paired Isaac↔MuJoCo evaluation are still in
  progress; they are deliberately not marked complete by a smoke test.

Run the contract checks without either simulator:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  backends/mujoco/tests/test_locomotion_contract.py
```

## Deployment boundary

The Python safety envelope is a reproducible research reference, not a
functional-safety certification.  Real hardware operation additionally needs
the vendor state machine, independent emergency stop, operator exclusion zone,
power/thermal monitoring and conservative first-motion procedures.
