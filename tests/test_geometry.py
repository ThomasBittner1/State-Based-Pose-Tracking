import cv2
import numpy as np

from src import geometry


def test_order_rectangle_points_returns_clockwise_from_top_left():
    points = np.array(
        [
            [40.0, 80.0],
            [10.0, 10.0],
            [45.0, 12.0],
            [12.0, 78.0],
        ],
        dtype=np.float32,
    )

    ordered = geometry.order_rectangle_points(points)

    np.testing.assert_allclose(
        ordered,
        np.array(
            [
                [10.0, 10.0],
                [45.0, 12.0],
                [40.0, 80.0],
                [12.0, 78.0],
            ],
            dtype=np.float32,
        ),
    )


def test_combine_detection_bounds_clips_to_frame():
    bounds = [(-5, 20, 50, 70), (10, -8, 120, 90)]

    combined = geometry.combine_detection_bounds(bounds, frame_shape=(80, 100, 3))

    assert combined == (0, 0, 100, 80)
    assert geometry.combine_detection_bounds([], frame_shape=(80, 100, 3)) is None
    assert geometry.combine_detection_bounds([(20, 10, 15, 30)], frame_shape=(80, 100, 3)) is None


def test_rate_homography_penalizes_bad_perspective_shear():
    identity = np.eye(3, dtype=np.float64)
    bad_homography = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.002, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert geometry.rate_homography(identity) == 1.0
    assert geometry.rate_homography(bad_homography) < 0.1



def test_pose_kalman_filter_disabled_and_enabled_behaviors():
    pose = (
        np.array([[0.1], [0.2], [0.3]], dtype=np.float64),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )

    filtered_pose = geometry.PoseKalmanFilter(enabled=False).filter_pose(pose)
    assert filtered_pose is pose

    rotation_vector, translation_vector = geometry.PoseKalmanFilter(enabled=True).filter_pose(pose)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    assert rotation_vector.shape == pose[0].shape
    assert translation_vector.shape == pose[1].shape
    assert np.isfinite(rotation_vector).all()
    assert np.isfinite(translation_vector).all()
    assert np.isclose(np.linalg.det(rotation_matrix), 1.0)
