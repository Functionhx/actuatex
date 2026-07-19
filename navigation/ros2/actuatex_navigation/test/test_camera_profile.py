from pathlib import Path

import pytest

from actuatex_navigation.camera_profile import (
    CameraCalibration,
    camera_frame_quaternions,
    load_camera_calibration,
    rotate_vector_wxyz,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _assert_vector(actual, expected):
    assert actual == pytest.approx(expected, abs=1.0e-9)


def test_load_example_camera_info():
    profile = load_camera_calibration(
        PACKAGE_ROOT / "config" / "front_camera_info.example.yaml"
    )
    assert profile.width == 640
    assert profile.height == 480
    assert profile.lens_schema == "OmniLensDistortionOpenCvPinholeAPI"
    assert profile.lens_model == "opencvPinhole"
    assert profile.lens_attributes()["omni:lensdistortion:opencvPinhole:fx"] == 600.0


def test_rational_polynomial_is_padded_to_full_schema():
    profile = CameraCalibration.from_mapping(
        {
            "image_width": 1280,
            "image_height": 720,
            "camera_matrix": {
                "data": [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
            },
            "distortion_model": "rational_polynomial",
            "distortion_coefficients": {
                "data": [0.1, -0.2, 0.001, -0.002, 0.03, 0.01, -0.01, 0.001]
            },
        }
    )
    attributes = profile.lens_attributes()
    assert attributes["omni:lensdistortion:opencvPinhole:k6"] == 0.001
    assert attributes["omni:lensdistortion:opencvPinhole:s4"] == 0.0


def test_accepts_direct_sensor_msgs_camera_info_fields():
    profile = CameraCalibration.from_mapping(
        {
            "width": 640,
            "height": 480,
            "k": [500.0, 0.0, 320.0, 0.0, 501.0, 240.0, 0.0, 0.0, 1.0],
            "distortion_model": "plumb_bob",
            "d": [0.1, -0.2, 0.001, -0.002, 0.03],
            "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "p": [
                500.0,
                0.0,
                320.0,
                0.0,
                0.0,
                501.0,
                240.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        }
    )
    assert profile.k[0] == 500.0
    assert profile.k[4] == 501.0
    assert profile.d[-1] == 0.03


@pytest.mark.parametrize(
    ("model", "coefficients"),
    [("plumb_bob", [0.0] * 4), ("equidistant", [0.0] * 5)],
)
def test_rejects_wrong_distortion_length(model, coefficients):
    with pytest.raises(ValueError, match="distortion coefficients"):
        CameraCalibration.from_mapping(
            {
                "image_width": 640,
                "image_height": 480,
                "camera_matrix": {
                    "data": [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
                },
                "distortion_model": model,
                "distortion_coefficients": {"data": coefficients},
            }
        )


def test_rejects_skew_that_renderer_cannot_reproduce():
    with pytest.raises(ValueError, match="skewed K"):
        CameraCalibration.from_mapping(
            {
                "image_width": 640,
                "image_height": 480,
                "camera_matrix": {
                    "data": [500.0, 1.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
                },
                "distortion_model": "plumb_bob",
                "distortion_coefficients": {"data": [0.0] * 5},
            }
        )


def test_forward_camera_axes_match_rep_103_and_ros_optical():
    usd_wxyz, ros_xyzw = camera_frame_quaternions((0.0, 0.0, 0.0))
    _assert_vector(rotate_vector_wxyz(usd_wxyz, (0.0, 0.0, -1.0)), (1.0, 0.0, 0.0))
    _assert_vector(rotate_vector_wxyz(usd_wxyz, (0.0, 1.0, 0.0)), (0.0, 0.0, 1.0))

    ros_wxyz = (ros_xyzw[3], ros_xyzw[0], ros_xyzw[1], ros_xyzw[2])
    _assert_vector(rotate_vector_wxyz(ros_wxyz, (0.0, 0.0, 1.0)), (1.0, 0.0, 0.0))
    _assert_vector(rotate_vector_wxyz(ros_wxyz, (0.0, 1.0, 0.0)), (0.0, 0.0, -1.0))


def test_positive_yaw_turns_camera_forward_to_robot_left():
    usd_wxyz, _ = camera_frame_quaternions((0.0, 0.0, 90.0))
    _assert_vector(rotate_vector_wxyz(usd_wxyz, (0.0, 0.0, -1.0)), (0.0, 1.0, 0.0))
