import cv2
import numpy as np


def draw_pose_axes_overlay(frame, camera_matrix, distortion_coefficients, pose_result, label_color, label=None):
    if not isinstance(pose_result, tuple) or len(pose_result) != 2:
        return

    rotation_vector, translation_vector = pose_result
    axis_points = np.array([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)], dtype=np.float32)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    axis_points_camera = (rotation_matrix @ axis_points.T).T + np.asarray(translation_vector, dtype=np.float64).reshape(1, 3)
    if not np.isfinite(axis_points_camera).all() or np.any(axis_points_camera[:, 2] <= 1e-6):
        return
    projected_points, _ = cv2.projectPoints(axis_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients)
    projected_points = projected_points.reshape(-1, 2)
    if not np.isfinite(projected_points).all():
        return
    projected_points = np.rint(projected_points).astype(np.int32)
    origin = tuple(projected_points[0])
    cv2.line(frame, origin, tuple(projected_points[1]), (0, 0, 255), 3)
    cv2.line(frame, origin, tuple(projected_points[2]), (0, 255, 0), 3)
    cv2.line(frame, origin, tuple(projected_points[3]), (255, 0, 0), 3)
    if label is not None:
        label_origin = (int(origin[0] + 6), int(origin[1] - 6))
        cv2.putText(frame, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2, cv2.LINE_AA)
        cv2.putText(frame, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
