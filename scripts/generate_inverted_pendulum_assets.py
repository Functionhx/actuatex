#!/usr/bin/env python3
"""Generate matched URDF and MJCF assets for serial cart-poles."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "robots" / "inverted_pendulum"

CART_MASS = 1.0
CART_SIZE = (0.30, 0.24, 0.16)
POLE_MASS = 0.20
POLE_LENGTH = 0.60
POLE_WIDTH = 0.04


def box_inertia(
    mass: float, size: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z = size
    return (
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def inertial_xml(
    mass: float, size: tuple[float, float, float], origin_z: float = 0.0
) -> str:
    ixx, iyy, izz = box_inertia(mass, size)
    return f"""    <inertial>
      <origin xyz="0 0 {origin_z:.6f}" rpy="0 0 0"/>
      <mass value="{mass:.8f}"/>
      <inertia ixx="{ixx:.10f}" ixy="0" ixz="0" iyy="{iyy:.10f}" iyz="0" izz="{izz:.10f}"/>
    </inertial>"""


def urdf(order: int) -> str:
    # Isaac Sim 6's URDF-to-USD pipeline mirrors source-link Z offsets in the
    # resulting fixed articulation.  Keep this asset intentionally expressed
    # with negative source Z so its imported world geometry matches the MJCF:
    # cart and pole centers of mass are above their parent joints at q = 0.
    pole_inertia = inertial_xml(
        POLE_MASS, (POLE_WIDTH, POLE_WIDTH, POLE_LENGTH), -POLE_LENGTH / 2
    )
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<robot name="actuatex_cartpole_{order}">',
        '  <material name="graphite"><color rgba="0.075 0.085 0.10 1"/></material>',
        '  <material name="orange"><color rgba="1.0 0.30 0.04 1"/></material>',
        '  <material name="metal"><color rgba="0.48 0.53 0.58 1"/></material>',
        '  <link name="rail">',
        inertial_xml(1.0, (5.2, 0.10, 0.04), -0.02),
        '    <visual><origin xyz="0 0 -0.02"/><geometry><box size="5.2 0.10 0.04"/></geometry><material name="metal"/></visual>',
        '    <collision><origin xyz="0 0 -0.02"/><geometry><box size="5.2 0.10 0.04"/></geometry></collision>',
        "  </link>",
        '  <link name="cart">',
        inertial_xml(CART_MASS, CART_SIZE),
        '    <visual><geometry><box size="0.30 0.24 0.16"/></geometry><material name="graphite"/></visual>',
        '    <collision><geometry><box size="0.30 0.24 0.16"/></geometry></collision>',
        "  </link>",
        '  <joint name="cart_slide" type="prismatic">',
        '    <parent link="rail"/><child link="cart"/>',
        '    <origin xyz="0 0 -0.16" rpy="0 0 0"/><axis xyz="1 0 0"/>',
        '    <limit lower="-2.4" upper="2.4" effort="20" velocity="8"/>',
        '    <dynamics damping="0.05" friction="0"/>',
        "  </joint>",
    ]
    for index in range(1, order + 1):
        material = "orange" if index % 2 else "metal"
        parent = "cart" if index == 1 else f"pole_{index - 1}"
        origin_z = -0.08 if index == 1 else -POLE_LENGTH
        lines.extend(
            [
                f'  <link name="pole_{index}">',
                pole_inertia,
                f'    <visual><origin xyz="0 0 {-POLE_LENGTH / 2:.3f}"/><geometry><box size="{POLE_WIDTH:.3f} {POLE_WIDTH:.3f} {POLE_LENGTH:.3f}"/></geometry><material name="{material}"/></visual>',
                f'    <collision><origin xyz="0 0 {-POLE_LENGTH / 2:.3f}"/><geometry><box size="{POLE_WIDTH:.3f} {POLE_WIDTH:.3f} {POLE_LENGTH:.3f}"/></geometry></collision>',
                "  </link>",
                f'  <joint name="pole_{index}_hinge" type="continuous">',
                f'    <parent link="{parent}"/><child link="pole_{index}"/>',
                f'    <origin xyz="0 0 {origin_z:.3f}" rpy="0 0 0"/><axis xyz="0 -1 0"/>',
                '    <limit effort="0" velocity="30"/>',
                '    <dynamics damping="0.002" friction="0"/>',
                "  </joint>",
            ]
        )
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def mjcf(order: int) -> str:
    cart_inertia = box_inertia(CART_MASS, CART_SIZE)
    pole_inertia = box_inertia(POLE_MASS, (POLE_WIDTH, POLE_WIDTH, POLE_LENGTH))
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<mujoco model="actuatex_cartpole_{order}">',
        '  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>',
        '  <option timestep="0.008333333333333333" gravity="0 0 -9.81" integrator="implicitfast" iterations="20"/>',
        '  <visual><global offwidth="1280" offheight="720"/></visual>',
        "  <default>",
        '    <joint armature="0.001"/>',
        '    <geom contype="0" conaffinity="0"/>',
        "  </default>",
        "  <asset>",
        '    <material name="graphite" rgba="0.075 0.085 0.10 1"/>',
        '    <material name="orange" rgba="1.0 0.30 0.04 1"/>',
        '    <material name="metal" rgba="0.48 0.53 0.58 1"/>',
        '    <material name="floor" rgba="0.14 0.16 0.19 1"/>',
        "  </asset>",
        "  <worldbody>",
        '    <light pos="0 -3 4" dir="0 0.4 -1" diffuse="0.9 0.9 0.9"/>',
        '    <geom name="ground" type="plane" size="0 0 0.1" material="floor"/>',
        '    <geom name="rail" type="box" pos="0 0 0.02" size="2.6 0.05 0.02" material="metal"/>',
        '    <body name="cart" pos="0 0 0.16">',
        '      <joint name="cart_slide" type="slide" axis="1 0 0" range="-2.4 2.4" damping="0.05"/>',
        f'      <inertial pos="0 0 0" mass="{CART_MASS:.8f}" diaginertia="{cart_inertia[0]:.10f} {cart_inertia[1]:.10f} {cart_inertia[2]:.10f}"/>',
        '      <geom name="cart_visual" type="box" size="0.15 0.12 0.08" material="graphite"/>',
    ]
    indentation = "      "
    for index in range(1, order + 1):
        material = "orange" if index % 2 else "metal"
        origin_z = 0.08 if index == 1 else POLE_LENGTH
        lines.extend(
            [
                f'{indentation}<body name="pole_{index}" pos="0 0 {origin_z:.3f}">',
                f'{indentation}  <joint name="pole_{index}_hinge" type="hinge" axis="0 1 0" damping="0.002"/>',
                f'{indentation}  <inertial pos="0 0 {POLE_LENGTH / 2:.3f}" mass="{POLE_MASS:.8f}" diaginertia="{pole_inertia[0]:.10f} {pole_inertia[1]:.10f} {pole_inertia[2]:.10f}"/>',
                f'{indentation}  <geom name="pole_{index}_visual" type="box" pos="0 0 {POLE_LENGTH / 2:.3f}" size="{POLE_WIDTH / 2:.3f} {POLE_WIDTH / 2:.3f} {POLE_LENGTH / 2:.3f}" material="{material}"/>',
            ]
        )
        indentation += "  "
    for _ in range(order):
        indentation = indentation[:-2]
        lines.append(f"{indentation}</body>")
    lines.extend(
        [
            "    </body>",
            "  </worldbody>",
            "  <actuator>",
            '    <motor name="cart_force" joint="cart_slide" gear="1" ctrllimited="true" ctrlrange="-20 20"/>',
            "  </actuator>",
            "</mujoco>",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    urdf_dir = args.output / "urdf"
    mjcf_dir = args.output / "mjcf"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    mjcf_dir.mkdir(parents=True, exist_ok=True)
    for order in (1, 2, 3):
        urdf_path = urdf_dir / f"actuatex_cartpole_{order}.urdf"
        mjcf_path = mjcf_dir / f"actuatex_cartpole_{order}.xml"
        urdf_path.write_text(urdf(order), encoding="utf-8")
        mjcf_path.write_text(mjcf(order), encoding="utf-8")
        print(urdf_path)
        print(mjcf_path)


if __name__ == "__main__":
    main()
