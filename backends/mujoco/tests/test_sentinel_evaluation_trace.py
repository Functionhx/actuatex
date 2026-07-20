from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.robomaster.evaluation_trace import (  # noqa: E402
    SentinelTraceRecorder,
    gravity_to_roll_pitch,
)


def test_gravity_to_roll_pitch_is_zero_when_upright() -> None:
    roll, pitch = gravity_to_roll_pitch(np.asarray([[0.0, 0.0, -1.0]]))
    np.testing.assert_allclose(roll, 0.0)
    np.testing.assert_allclose(pitch, 0.0)


def test_trace_records_posture_action_power_and_falls(tmp_path: Path) -> None:
    recorder = SentinelTraceRecorder(2)
    recorder.record(
        time_s=0.02,
        segment="stand",
        command=(0.0, 0.0, 0.0),
        base_linear_velocity_body=np.zeros((2, 3)),
        base_angular_velocity_body=np.zeros((2, 3)),
        projected_gravity=np.asarray([[0.0, 0.0, -1.0]] * 2),
        base_height_m=np.asarray([0.51, 0.50]),
        action=np.asarray([[0.0] * 6, [1.0] * 6]),
        chassis_power_w=np.asarray([40.0, 60.0]),
        motor_temperature_c=np.full((2, 11), 26.0),
        buffer_energy_j=np.asarray([58.0, 59.0]),
        terminal=np.asarray([False, True]),
    )
    row = recorder.rows[0]
    assert row["base_height_min_m"] == pytest.approx(0.50)
    assert row["action_saturation_fraction"] == pytest.approx(0.5)
    assert row["chassis_power_mean_w"] == pytest.approx(50.0)
    assert row["fall_events_cumulative"] == 1
    assert row["envs_ever_fallen"] == 1

    output = tmp_path / "trace.csv"
    recorder.write_csv(output)
    assert output.read_text(encoding="utf-8").startswith("time_s,segment,")

    svg = tmp_path / "trace.svg"
    recorder.write_svg(svg, title="Sentinel test")
    rendered = svg.read_text(encoding="utf-8")
    assert rendered.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Sentinel test" in rendered
