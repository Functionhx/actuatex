# Architecture

ActuateX separates portable experiment logic from simulator runtimes:

```text
                     ┌──────────────────────┐
                     │ TinyMal contract     │
                     │ URDF / joints / obs  │
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

The contract is more important than the network file format. A transferred actor is only meaningful when every backend agrees on joint order, coordinate frames, observation scaling, action scale, policy interval, actuator limits, reset state, and metric definitions.

Upstream projects live in ignored `_deps/` checkouts. `upstream.json` files pin compatible revisions, while guarded installers apply the patches and overlays. Runtime products live under ignored `artifacts/`, so Git history contains reviewable source rather than checkpoints and simulator binaries.
