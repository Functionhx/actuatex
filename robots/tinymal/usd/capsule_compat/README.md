# TinyMal capsule-compatible USD

This asset preserves the collision geometry used by the original Isaac Gym
and Isaac Lab 5.1 experiments after Isaac Sim 6 removed the URDF importer
option `replace_cylinders_with_capsules`.

- Source: `robots/tinymal/urdf/tinymal.urdf`
- Generator: Isaac Sim 5.1 URDF importer `2.4.30`
- Import setting: `replace_cylinders_with_capsules=true`
- Result: four hip colliders are USD capsules with radius `0.03 m`, height
  `0.03 m`, Z axis, and `UsdPhysics.CollisionAPI`
- Portability check: the composed stage has no absolute external references
- Integrity: see `manifest.json`

Isaac Sim 6.0.1 training uses `tinymal.usd` by default. To deliberately test
the new native-cylinder importer instead, set `ACTUATEX_TINYMAL_URDF` to the
source URDF before launching Isaac Lab. This opt-in is an A/B experiment, not
the migration baseline.
