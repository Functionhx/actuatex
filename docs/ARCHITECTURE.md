# Architecture

ActuateX separates portable robot contracts from simulator, training and
deployment runtimes.  TinyMal remains the teaching baseline; wheel-legged and
G1 humanoid tasks use the same boundary:

```text
                     ┌──────────────────────┐
                     │ Robot contract       │
                     │ asset / joints / obs │
                     │ action / timing / PD │
                     └──────────┬───────────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
        ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
        │ Isaac Gym   │  │ Isaac Lab   │  │ MuJoCo      │
        │ PhysX base  │  │ PhysX 5     │  │ native PPO  │
        └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
               │                │                │
               └────────────────┼────────────────┘
                                ▼
                     shared acceptance suite
               tracking / falls / stairs / pushes
```

The contract is more important than the network file format. A transferred actor is only meaningful when every backend agrees on joint order, coordinate frames, observation scaling and history layout, action scale, policy interval, actuator limits, reset state, and metric definitions. The same contract also travels beside an exported ONNX model into the C++/DDS runtime.

The complete robotics stack adds state estimation and ROS 2 navigation above
the locomotion actor, and an independent safety supervisor below it.  See the
[industrial RL stack](./INDUSTRIAL_RL_STACK.zh-CN.md) for the quadruped and G1
humanoid plan, current evidence, and explicit completion gates.

Upstream projects live in ignored `_deps/` checkouts. `upstream.json` files pin compatible revisions, while guarded installers apply the patches and overlays. Runtime products live under ignored `artifacts/`, so Git history contains reviewable source rather than checkpoints and simulator binaries.
