from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from actuatex_navigation.mid360_pattern import (
    CHANNEL_COUNT,
    FRAME_COUNT,
    FRAME_PERIOD_NS,
    PATTERN_CYCLE_NS,
    POINT_COUNT,
    POINT_PERIOD_NS,
    POINTS_PER_FRAME,
    load_mid360_pattern,
)


@lru_cache(maxsize=1)
def _pattern():
    return load_mid360_pattern()


def test_official_pattern_dimensions_and_coordinates() -> None:
    pattern = _pattern()

    assert pattern.point_count == POINT_COUNT == 800_000
    assert pattern.frame_count == FRAME_COUNT == 40
    assert pattern.cycle_seconds == pytest.approx(4.0)
    assert PATTERN_CYCLE_NS == 4_000_000_000
    np.testing.assert_allclose(
        [pattern.azimuth_deg[0], pattern.elevation_deg[0]],
        [-91.01, 52.162],
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        [pattern.azimuth_deg[-1], pattern.elevation_deg[-1]],
        [33.069, 49.939],
        atol=1.0e-5,
    )
    assert float(pattern.elevation_deg.min()) == pytest.approx(-7.2123, abs=1.0e-5)
    assert float(pattern.elevation_deg.max()) == pytest.approx(52.164, abs=1.0e-5)


def test_frame_preserves_point_timing_and_four_channels() -> None:
    frame = _pattern().frame(0)

    assert frame.point_count == POINTS_PER_FRAME == 20_000
    assert POINT_PERIOD_NS == 5_000
    assert FRAME_PERIOD_NS == 100_000_000
    assert frame.fire_time_ns[0] == 0
    assert frame.fire_time_ns[-1] == FRAME_PERIOD_NS - POINT_PERIOD_NS
    np.testing.assert_array_equal(
        frame.channel_id[: 2 * CHANNEL_COUNT],
        [0, 1, 2, 3, 0, 1, 2, 3],
    )


def test_explicit_stride_reduces_load_without_changing_acquisition_time() -> None:
    frame = _pattern().frame(FRAME_COUNT - 1, stride=4)

    assert frame.point_count == POINTS_PER_FRAME // 4
    assert frame.fire_time_ns[1] == 4 * POINT_PERIOD_NS
    assert frame.fire_time_ns[-1] == FRAME_PERIOD_NS - 4 * POINT_PERIOD_NS
    assert np.all(frame.channel_id == 0)


@pytest.mark.parametrize("index", [-1, FRAME_COUNT])
def test_invalid_frame_index_is_rejected(index: int) -> None:
    with pytest.raises(IndexError):
        _pattern().frame(index)
