import math

import pytest

from actuatex_navigation.command_filter import (
    CommandFilter,
    VelocityCommand,
    VelocityLimits,
)


def test_target_is_clamped_to_policy_envelope():
    command_filter = CommandFilter()
    limited = command_filter.limit_target(VelocityCommand(2.0, -1.0, 4.0))
    assert limited == VelocityCommand(0.55, -0.25, 0.75)


def test_reverse_limit_is_independent_from_forward_limit():
    command_filter = CommandFilter()
    assert command_filter.limit_target(VelocityCommand(x=-1.0)).x == -0.25
    assert command_filter.limit_target(VelocityCommand(x=1.0)).x == 0.55


def test_holonomic_lateral_command_is_preserved():
    command_filter = CommandFilter()
    limited = command_filter.limit_target(VelocityCommand(y=0.18))
    assert limited.y == pytest.approx(0.18)


def test_slew_rate_is_applied_per_axis():
    limits = VelocityLimits(
        acceleration_x=0.8,
        acceleration_y=0.6,
        acceleration_yaw=1.2,
    )
    command_filter = CommandFilter(limits)
    command = command_filter.update(VelocityCommand(0.5, 0.2, 0.5), dt=0.1)
    assert command == VelocityCommand(
        pytest.approx(0.08), pytest.approx(0.06), pytest.approx(0.12)
    )


def test_deadband_and_non_finite_input_fail_safe_to_zero():
    command_filter = CommandFilter()
    assert (
        command_filter.limit_target(VelocityCommand(0.01, -0.01, 0.03))
        == VelocityCommand.zero()
    )
    assert (
        command_filter.limit_target(VelocityCommand(math.nan, 0.1, 0.1))
        == VelocityCommand.zero()
    )


def test_watchdog_stop_is_immediate():
    command_filter = CommandFilter()
    command_filter.update(VelocityCommand(x=0.5), dt=0.2)
    assert command_filter.current.x > 0.0
    assert command_filter.stop() == VelocityCommand.zero()


@pytest.mark.parametrize("dt", [0.0, -0.1, math.inf, math.nan])
def test_invalid_time_step_is_rejected(dt):
    with pytest.raises(ValueError):
        CommandFilter().update(VelocityCommand(), dt)
