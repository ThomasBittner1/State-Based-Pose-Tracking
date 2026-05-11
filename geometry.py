import cv2
import numpy as np


def combine_detection_bounds(bounds_history, frame_shape):
    if not bounds_history:
        return None

    frame_height, frame_width = frame_shape[:2]
    left = max(0, min(bounds[0] for bounds in bounds_history))
    top = max(0, min(bounds[1] for bounds in bounds_history))
    right = min(frame_width, max(bounds[2] for bounds in bounds_history))
    bottom = min(frame_height, max(bounds[3] for bounds in bounds_history))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def blend_pose_result(previous_pose_result, latest_pose_result, pose_blend):
    if previous_pose_result is None or latest_pose_result is None:
        return latest_pose_result
    previous_rotation_vector, previous_translation_vector = previous_pose_result
    latest_rotation_vector, latest_translation_vector = latest_pose_result
    blended_rotation_vector = previous_rotation_vector * (1.0 - pose_blend) + latest_rotation_vector * pose_blend
    blended_translation_vector = previous_translation_vector * (1.0 - pose_blend) + latest_translation_vector * pose_blend
    return blended_rotation_vector, blended_translation_vector


def rotation_matrix_similarity(rotation_matrix_a, rotation_matrix_b):
    rotation_matrix_a = np.asarray(rotation_matrix_a, dtype=np.float64)
    rotation_matrix_b = np.asarray(rotation_matrix_b, dtype=np.float64)
    if rotation_matrix_a.shape != (3, 3) or rotation_matrix_b.shape != (3, 3):
        raise ValueError("rotation matrices must both be 3x3")

    axis_scores = []
    for axis_index in range(3):
        axis_a = rotation_matrix_a[:, axis_index]
        axis_b = rotation_matrix_b[:, axis_index]
        axis_a_norm = np.linalg.norm(axis_a)
        axis_b_norm = np.linalg.norm(axis_b)
        if axis_a_norm <= 1e-8 or axis_b_norm <= 1e-8:
            raise ValueError("rotation matrix axes must be non-zero")
        axis_dot = float(np.dot(axis_a / axis_a_norm, axis_b / axis_b_norm))
        axis_scores.append((np.clip(axis_dot, -1.0, 1.0) + 1.0) * 0.5)
    return float(min(axis_scores))


def rate_homography(homography):
    if homography is None:
        return 0.0

    last_row = homography[2]
    x_axis = np.copy(homography[0:2,0])
    y_axis = np.copy(homography[0:2,1])
    x_length = np.linalg.norm(x_axis)
    y_length = np.linalg.norm(y_axis)

    x_axis /= x_length
    y_axis /= y_length
    dot_product = np.sum(x_axis * y_axis)
    score_dot_product = np.interp(abs(dot_product), [0.0, 0.3, 0.75], [1.0, 1.0, 0.0])
    score_shear_0 = np.interp(abs(last_row[0]), [0.00099, 0.001], [1.0, 0.0])
    score_shear_1 = np.interp(abs(last_row[1]), [0.00099, 0.001], [1.0, 0.0])
    mean_score = score_dot_product * score_shear_0 * score_shear_1
    return mean_score


class PoseKalmanFilter:
    def __init__(self, enabled=True, translation_process_noise=1e-4, translation_measurement_noise=1e-2,
                 rotation_process_noise=1e-4, rotation_measurement_noise=1e-2):
        self.enabled = bool(enabled)
        self.translation_process_noise = float(translation_process_noise)
        self.translation_measurement_noise = float(translation_measurement_noise)
        self.rotation_process_noise = float(rotation_process_noise)
        self.rotation_measurement_noise = float(rotation_measurement_noise)
        self.reset()

    def _create_constant_velocity_filter(self, measurement_dim, process_noise, measurement_noise):
        state_dim = measurement_dim * 2
        kalman = cv2.KalmanFilter(state_dim, measurement_dim, 0, cv2.CV_64F)
        kalman.transitionMatrix = np.eye(state_dim, dtype=np.float64)
        kalman.transitionMatrix[:measurement_dim, measurement_dim:] = np.eye(measurement_dim, dtype=np.float64)
        kalman.measurementMatrix = np.zeros((measurement_dim, state_dim), dtype=np.float64)
        kalman.measurementMatrix[:, :measurement_dim] = np.eye(measurement_dim, dtype=np.float64)
        kalman.processNoiseCov = np.eye(state_dim, dtype=np.float64) * process_noise
        kalman.measurementNoiseCov = np.eye(measurement_dim, dtype=np.float64) * measurement_noise
        kalman.errorCovPost = np.eye(state_dim, dtype=np.float64)
        kalman.statePost = np.zeros((state_dim, 1), dtype=np.float64)
        kalman.statePre = np.zeros((state_dim, 1), dtype=np.float64)
        return kalman

    def reset(self):
        self.translation_filter = self._create_constant_velocity_filter(
            3, self.translation_process_noise, self.translation_measurement_noise
        )
        self.rotation_filter = self._create_constant_velocity_filter(
            9, self.rotation_process_noise, self.rotation_measurement_noise
        )
        self.is_initialized = False

    def _initialize_filter(self, kalman, measurement):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(-1)
        kalman.statePost[:measurement.size, 0] = measurement
        kalman.statePre[:measurement.size, 0] = measurement
        kalman.statePost[measurement.size:, 0] = 0.0
        kalman.statePre[measurement.size:, 0] = 0.0

    def _update_filter(self, kalman, measurement):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(-1, 1)
        kalman.predict()
        filtered_state = kalman.correct(measurement)
        return filtered_state[:measurement.shape[0], 0]

    def filter_pose(self, pose_result):
        if not self.enabled or pose_result is None:
            return pose_result

        rotation_vector, translation_vector = pose_result
        rotation_vector = np.asarray(rotation_vector, dtype=np.float64)
        translation_vector = np.asarray(translation_vector, dtype=np.float64).reshape(3)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        if not self.is_initialized:
            self._initialize_filter(self.translation_filter, translation_vector)
            self._initialize_filter(self.rotation_filter, rotation_matrix.reshape(-1))
            self.is_initialized = True

        filtered_translation = self._update_filter(self.translation_filter, translation_vector)
        filtered_rotation = self._update_filter(self.rotation_filter, rotation_matrix.reshape(-1)).reshape(3, 3)
        rotation_u, _, rotation_vt = np.linalg.svd(filtered_rotation)
        filtered_rotation = rotation_u @ rotation_vt
        if np.linalg.det(filtered_rotation) < 0.0:
            rotation_u[:, -1] *= -1.0
            filtered_rotation = rotation_u @ rotation_vt

        filtered_rotation_vector, _ = cv2.Rodrigues(filtered_rotation)
        return filtered_rotation_vector.reshape(rotation_vector.shape), filtered_translation.reshape(np.asarray(translation_vector).shape)


def order_rectangle_points(rectangle_points):
    sums = rectangle_points.sum(axis=1)
    diffs = np.diff(rectangle_points, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = rectangle_points[np.argmin(sums)]
    ordered[2] = rectangle_points[np.argmax(sums)]
    ordered[1] = rectangle_points[np.argmin(diffs)]
    ordered[3] = rectangle_points[np.argmax(diffs)]
    return ordered


def is_face_visible(face_points, face_points_camera):
    p0, p1, p2 = face_points[:3]
    normal_box = np.cross(p1 - p0, p2 - p0)
    face_center_box = np.mean(face_points, axis=0)
    p0_camera, p1_camera, p2_camera = face_points_camera[:3]
    normal = np.cross(p1_camera - p0_camera, p2_camera - p0_camera)
    if np.dot(normal_box, face_center_box) < 0.0:
        normal = -normal
    face_center = np.mean(face_points_camera, axis=0)
    return np.dot(normal, face_center) < 0.0
