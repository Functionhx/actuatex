"""Shared time-series evidence for Sentinel controller evaluations."""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .contract import ACTION_DIM


TRACE_FIELD_NAMES = (
    "time_s",
    "segment",
    "command_vx_mps",
    "command_vy_mps",
    "command_yaw_radps",
    "actual_vx_mean_mps",
    "actual_vx_p10_mps",
    "actual_vx_p90_mps",
    "actual_vy_mean_mps",
    "actual_yaw_mean_radps",
    "actual_yaw_p10_radps",
    "actual_yaw_p90_radps",
    "roll_mean_deg",
    "roll_p95_abs_deg",
    "pitch_mean_deg",
    "pitch_p95_abs_deg",
    "base_height_mean_m",
    "base_height_min_m",
    "action_abs_mean",
    "leg_action_abs_mean",
    "wheel_action_abs_mean",
    "action_saturation_fraction",
    "chassis_power_mean_w",
    "chassis_power_max_w",
    "motor_temperature_mean_c",
    "motor_temperature_max_c",
    "buffer_energy_min_j",
    "fall_events_step",
    "fall_events_cumulative",
    "envs_ever_fallen",
) + tuple(f"action_{index}_mean" for index in range(ACTION_DIM))


def gravity_to_roll_pitch(projected_gravity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover gravity-referenced roll/pitch without relying on global yaw."""

    gravity = np.asarray(projected_gravity, dtype=np.float64)
    if gravity.ndim != 2 or gravity.shape[1] != 3:
        raise ValueError("projected gravity must have shape (N, 3)")
    if not np.isfinite(gravity).all():
        raise ValueError("projected gravity contains a non-finite value")
    roll = np.arctan2(-gravity[:, 1], -gravity[:, 2])
    pitch = np.arctan2(
        gravity[:, 0],
        np.sqrt(np.square(gravity[:, 1]) + np.square(gravity[:, 2])),
    )
    return roll, pitch


class SentinelTraceRecorder:
    """Aggregate a vectorized rollout into one plot-friendly row per step."""

    def __init__(self, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError("trace recorder needs at least one environment")
        self.num_envs = int(num_envs)
        self.rows: list[dict[str, Any]] = []
        self._ever_fallen = np.zeros(self.num_envs, dtype=bool)
        self._fall_events = 0

    def record(
        self,
        *,
        time_s: float,
        segment: str,
        command: tuple[float, float, float],
        base_linear_velocity_body: np.ndarray,
        base_angular_velocity_body: np.ndarray,
        projected_gravity: np.ndarray,
        base_height_m: np.ndarray,
        action: np.ndarray,
        chassis_power_w: np.ndarray,
        motor_temperature_c: np.ndarray,
        buffer_energy_j: np.ndarray,
        terminal: np.ndarray,
    ) -> None:
        linear_velocity = np.asarray(base_linear_velocity_body, dtype=np.float64)
        angular_velocity = np.asarray(
            base_angular_velocity_body, dtype=np.float64
        )
        gravity = np.asarray(projected_gravity, dtype=np.float64)
        height = np.asarray(base_height_m, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        power = np.asarray(chassis_power_w, dtype=np.float64)
        temperature = np.asarray(motor_temperature_c, dtype=np.float64)
        buffer = np.asarray(buffer_energy_j, dtype=np.float64)
        terminal = np.asarray(terminal, dtype=bool)
        expected_vector = (self.num_envs,)
        if linear_velocity.shape != (self.num_envs, 3):
            raise ValueError("base linear velocity trace shape mismatch")
        if angular_velocity.shape != (self.num_envs, 3):
            raise ValueError("base angular velocity trace shape mismatch")
        if gravity.shape != (self.num_envs, 3):
            raise ValueError("projected gravity trace shape mismatch")
        if action.shape != (self.num_envs, ACTION_DIM):
            raise ValueError("action trace shape mismatch")
        if any(
            values.shape != expected_vector
            for values in (height, power, buffer, terminal)
        ):
            raise ValueError("scalar-per-environment trace shape mismatch")
        if temperature.ndim != 2 or temperature.shape[0] != self.num_envs:
            raise ValueError("motor temperature trace shape mismatch")
        numeric = (
            linear_velocity,
            angular_velocity,
            gravity,
            height,
            action,
            power,
            temperature,
            buffer,
        )
        if not all(np.isfinite(values).all() for values in numeric):
            raise ValueError("trace contains a non-finite value")

        roll, pitch = gravity_to_roll_pitch(gravity)
        fall_events = int(np.count_nonzero(terminal))
        self._fall_events += fall_events
        self._ever_fallen |= terminal
        action_mean = np.mean(action, axis=0)
        row: dict[str, Any] = {
            "time_s": float(time_s),
            "segment": segment,
            "command_vx_mps": float(command[0]),
            "command_vy_mps": float(command[1]),
            "command_yaw_radps": float(command[2]),
            "actual_vx_mean_mps": float(np.mean(linear_velocity[:, 0])),
            "actual_vx_p10_mps": float(np.quantile(linear_velocity[:, 0], 0.10)),
            "actual_vx_p90_mps": float(np.quantile(linear_velocity[:, 0], 0.90)),
            "actual_vy_mean_mps": float(np.mean(linear_velocity[:, 1])),
            "actual_yaw_mean_radps": float(np.mean(angular_velocity[:, 2])),
            "actual_yaw_p10_radps": float(np.quantile(angular_velocity[:, 2], 0.10)),
            "actual_yaw_p90_radps": float(np.quantile(angular_velocity[:, 2], 0.90)),
            "roll_mean_deg": float(np.degrees(np.mean(roll))),
            "roll_p95_abs_deg": float(np.degrees(np.quantile(np.abs(roll), 0.95))),
            "pitch_mean_deg": float(np.degrees(np.mean(pitch))),
            "pitch_p95_abs_deg": float(
                np.degrees(np.quantile(np.abs(pitch), 0.95))
            ),
            "base_height_mean_m": float(np.mean(height)),
            "base_height_min_m": float(np.min(height)),
            "action_abs_mean": float(np.mean(np.abs(action))),
            "leg_action_abs_mean": float(np.mean(np.abs(action[:, :4]))),
            "wheel_action_abs_mean": float(np.mean(np.abs(action[:, 4:]))),
            "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.999)),
            "chassis_power_mean_w": float(np.mean(power)),
            "chassis_power_max_w": float(np.max(power)),
            "motor_temperature_mean_c": float(np.mean(temperature)),
            "motor_temperature_max_c": float(np.max(temperature)),
            "buffer_energy_min_j": float(np.min(buffer)),
            "fall_events_step": fall_events,
            "fall_events_cumulative": self._fall_events,
            "envs_ever_fallen": int(np.count_nonzero(self._ever_fallen)),
        }
        row.update(
            {
                f"action_{index}_mean": float(action_mean[index])
                for index in range(ACTION_DIM)
            }
        )
        self.rows.append(row)

    def write_csv(self, path: Path) -> None:
        if not self.rows:
            raise RuntimeError("cannot write an empty Sentinel trace")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACE_FIELD_NAMES)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_svg(self, path: Path, *, title: str) -> None:
        if not self.rows:
            raise RuntimeError("cannot plot an empty Sentinel trace")

        def column(name: str) -> np.ndarray:
            return np.asarray([row[name] for row in self.rows], dtype=np.float64)

        time_s = column("time_s")
        panels = (
            (
                "vx [m/s]",
                (
                    ("command_vx_mps", "target", "#111827", True),
                    ("actual_vx_mean_mps", "mean", "#2563eb", False),
                    ("actual_vx_p10_mps", "p10", "#93c5fd", True),
                    ("actual_vx_p90_mps", "p90", "#60a5fa", True),
                ),
            ),
            (
                "yaw rate [rad/s]",
                (
                    ("command_yaw_radps", "target", "#111827", True),
                    ("actual_yaw_mean_radps", "mean", "#7c3aed", False),
                    ("actual_yaw_p10_radps", "p10", "#c4b5fd", True),
                    ("actual_yaw_p90_radps", "p90", "#a78bfa", True),
                ),
            ),
            (
                "attitude [deg]",
                (
                    ("roll_mean_deg", "roll mean", "#059669", False),
                    ("pitch_mean_deg", "pitch mean", "#dc2626", False),
                    ("pitch_p95_abs_deg", "|pitch| p95", "#fca5a5", True),
                ),
            ),
            (
                "base height [m]",
                (
                    ("base_height_mean_m", "mean", "#0891b2", False),
                    ("base_height_min_m", "minimum", "#67e8f9", True),
                ),
            ),
            (
                "normalized action",
                (
                    ("leg_action_abs_mean", "leg |u|", "#d97706", False),
                    ("wheel_action_abs_mean", "wheel |u|", "#ea580c", False),
                    (
                        "action_saturation_fraction",
                        "saturation",
                        "#991b1b",
                        True,
                    ),
                ),
            ),
            (
                "chassis power [W]",
                (
                    ("chassis_power_mean_w", "mean", "#0f766e", False),
                    ("chassis_power_max_w", "maximum", "#5eead4", True),
                ),
            ),
            (
                "motor temperature [C]",
                (
                    ("motor_temperature_mean_c", "mean", "#be123c", False),
                    ("motor_temperature_max_c", "maximum", "#fb7185", True),
                ),
            ),
            (
                "cumulative fall events",
                (("fall_events_cumulative", "falls", "#b91c1c", False),),
            ),
        )
        width = 1400
        left = 125
        right = 35
        top = 80
        panel_height = 145
        panel_gap = 38
        bottom = 65
        plot_width = width - left - right
        height = top + len(panels) * (panel_height + panel_gap) + bottom
        time_min = float(time_s[0])
        time_max = float(time_s[-1])
        if time_max <= time_min:
            time_max = time_min + 1.0

        def x_coordinate(value: float) -> float:
            return left + (value - time_min) / (time_max - time_min) * plot_width

        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            "<style>text{font-family:Inter,DejaVu Sans,sans-serif;fill:#1f2937}"
            ".tick{font-size:12px;fill:#4b5563}.label{font-size:14px;font-weight:600}"
            ".legend{font-size:12px}.grid{stroke:#e5e7eb;stroke-width:1}"
            ".axis{stroke:#6b7280;stroke-width:1.2}</style>",
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'<text x="{width / 2}" y="38" text-anchor="middle" '
            f'font-size="22" font-weight="700">{escape(title)}</text>',
        ]
        segment_boundaries = []
        previous_segment = self.rows[0]["segment"]
        segment_boundaries.append((time_s[0], previous_segment))
        for index, row in enumerate(self.rows[1:], start=1):
            if row["segment"] != previous_segment:
                segment_boundaries.append((time_s[index], row["segment"]))
                previous_segment = row["segment"]

        for panel_index, (axis_label, series) in enumerate(panels):
            panel_top = top + panel_index * (panel_height + panel_gap)
            panel_bottom = panel_top + panel_height
            values = np.concatenate([column(name) for name, *_ in series])
            value_min = min(float(np.min(values)), 0.0)
            value_max = max(float(np.max(values)), 0.0)
            if value_max - value_min < 1.0e-9:
                value_min -= 0.5
                value_max += 0.5
            padding = 0.08 * (value_max - value_min)
            value_min -= padding
            value_max += padding

            def y_coordinate(value: float) -> float:
                ratio = (value - value_min) / (value_max - value_min)
                return panel_bottom - ratio * panel_height

            for grid_index in range(5):
                fraction = grid_index / 4
                y = panel_bottom - fraction * panel_height
                tick_value = value_min + fraction * (value_max - value_min)
                elements.append(
                    f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                    f'x2="{left + plot_width}" y2="{y:.2f}"/>'
                )
                elements.append(
                    f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" '
                    f'text-anchor="end">{tick_value:.3g}</text>'
                )
            elements.append(
                f'<line class="axis" x1="{left}" y1="{panel_top}" '
                f'x2="{left}" y2="{panel_bottom}"/>'
            )
            elements.append(
                f'<text class="label" transform="translate(25 '
                f'{(panel_top + panel_bottom) / 2:.2f}) rotate(-90)" '
                f'text-anchor="middle">{escape(axis_label)}</text>'
            )
            for boundary_time, segment_name in segment_boundaries:
                x = x_coordinate(float(boundary_time))
                elements.append(
                    f'<line x1="{x:.2f}" y1="{panel_top}" x2="{x:.2f}" '
                    f'y2="{panel_bottom}" stroke="#d1d5db" stroke-width="1"/>'
                )
                if panel_index == 0:
                    elements.append(
                        f'<text class="tick" x="{x + 4:.2f}" '
                        f'y="{panel_top - 7}">{escape(str(segment_name))}</text>'
                    )
            legend_x = left + 8
            for series_index, (name, label, color, dashed) in enumerate(series):
                values = column(name)
                points = " ".join(
                    f"{x_coordinate(float(x)):.2f},{y_coordinate(float(y)):.2f}"
                    for x, y in zip(time_s, values, strict=True)
                )
                dash = ' stroke-dasharray="7,5"' if dashed else ""
                elements.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    f'stroke-width="2"{dash}/>'
                )
                item_x = legend_x + series_index * 160
                elements.append(
                    f'<line x1="{item_x}" y1="{panel_top + 14}" '
                    f'x2="{item_x + 24}" y2="{panel_top + 14}" '
                    f'stroke="{color}" stroke-width="3"{dash}/>'
                )
                elements.append(
                    f'<text class="legend" x="{item_x + 30}" '
                    f'y="{panel_top + 18}">{escape(label)}</text>'
                )

        final_bottom = top + (len(panels) - 1) * (
            panel_height + panel_gap
        ) + panel_height
        for tick_index in range(6):
            fraction = tick_index / 5
            tick_time = time_min + fraction * (time_max - time_min)
            x = x_coordinate(tick_time)
            elements.append(
                f'<line class="axis" x1="{x:.2f}" y1="{final_bottom}" '
                f'x2="{x:.2f}" y2="{final_bottom + 6}"/>'
            )
            elements.append(
                f'<text class="tick" x="{x:.2f}" y="{final_bottom + 23}" '
                f'text-anchor="middle">{tick_time:.1f}</text>'
            )
        elements.append(
            f'<text class="label" x="{left + plot_width / 2}" '
            f'y="{final_bottom + 48}" text-anchor="middle">time [s]</text>'
        )
        elements.append("</svg>")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(elements) + "\n", encoding="utf-8")
