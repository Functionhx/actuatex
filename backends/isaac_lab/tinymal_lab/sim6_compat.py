"""Small runtime compatibility fixes for Isaac Sim 6 generated assets."""

from __future__ import annotations


def enable_nested_rigid_body_contact_sensors() -> None:
    """Teach Isaac Lab's contact helper about Sim 6 nested link prims.

    Isaac Sim 6's URDF asset transformer represents a kinematic chain as nested
    rigid-body prims.  Isaac Lab 3.0's helper stops walking as soon as it sees
    the root rigid body, based on the older assumption that rigid bodies cannot
    be nested.  Consequently only ``base_link`` receives
    ``PhysxContactReportAPI``.  This wrapper preserves the upstream helper but
    invokes it for every rigid descendant discovered before traversal stops.

    The patch is process-local, idempotent, and only installed when the wheel
    task is imported; it does not modify the pinned Isaac Lab checkout.
    """
    from pxr import UsdPhysics

    from isaaclab.sim.schemas import schemas
    from isaaclab.sim.utils import get_current_stage

    current = schemas.activate_contact_sensors
    if getattr(current, "_actuatex_nested_body_compat", False):
        return

    upstream = current

    def activate_contact_sensors_nested(
        prim_path: str, threshold: float = 0.0, stage=None
    ):
        if stage is None:
            stage = get_current_stage()
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            return upstream(prim_path, threshold=threshold, stage=stage)

        rigid_paths: list[str] = []
        queue = [root]
        while queue:
            prim = queue.pop(0)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_paths.append(prim.GetPath().pathString)
            # Sim 6 permits rigid link prims below another rigid link prim.
            queue.extend(prim.GetChildren())

        if not rigid_paths:
            return upstream(prim_path, threshold=threshold, stage=stage)
        for rigid_path in rigid_paths:
            upstream(rigid_path, threshold=threshold, stage=stage)
        return True

    activate_contact_sensors_nested._actuatex_nested_body_compat = True
    schemas.activate_contact_sensors = activate_contact_sensors_nested
