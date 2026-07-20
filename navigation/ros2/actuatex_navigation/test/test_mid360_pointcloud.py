from __future__ import annotations

import numpy as np
import pytest

from actuatex_navigation.mid360_pointcloud import (
    LIVOX_POINT_DTYPE,
    LIVOX_POINT_STEP,
    POINT_FIELDS,
    build_livox_points,
    project_planar_scan,
)


def test_driver_point_layout_and_absolute_timestamp() -> None:
    points = build_livox_points(
        xyz=np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        intensity=np.asarray((12.5, 300.0)),
        timebase_ns=1_000_000_000,
        offset_time_ns=np.asarray((0, 5_000)),
        line=np.asarray((0, 3)),
    )

    assert points.dtype == LIVOX_POINT_DTYPE
    assert points.dtype.itemsize == LIVOX_POINT_STEP == 26
    assert len(points.tobytes()) == 2 * LIVOX_POINT_STEP
    assert POINT_FIELDS[-1] == ("timestamp", 18, 8)
    np.testing.assert_allclose(points["intensity"], [12.5, 255.0])
    np.testing.assert_array_equal(points["tag"], [0, 0])
    np.testing.assert_array_equal(points["line"], [0, 3])
    np.testing.assert_allclose(points["timestamp"], [1.0e9, 1.000005e9])


def test_height_band_projection_uses_nearest_return_per_bin() -> None:
    xyz = np.asarray(
        (
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (-2.0, 0.0, 0.3),
        ),
        dtype=np.float32,
    )
    projection = project_planar_scan(xyz, bins=360, min_z=-0.1, max_z=0.1)

    forward_index = 180
    left_index = 270
    assert projection.ranges[forward_index] == pytest.approx(1.0)
    assert projection.ranges[left_index] == pytest.approx(2.0)
    assert np.isinf(projection.ranges[0])
    assert projection.angle_increment == pytest.approx(2.0 * np.pi / 360)


def test_negative_point_offset_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_livox_points(
            xyz=np.zeros((1, 3)),
            intensity=np.zeros(1),
            timebase_ns=0,
            offset_time_ns=np.asarray((-1,)),
            line=np.zeros(1),
        )
