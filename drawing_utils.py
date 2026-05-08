import cv2
import numpy as np


def draw_box_overlay(frame, camera_matrix, distortion_coefficients, pose_result, box_size, opacity=1.0):
    if not isinstance(pose_result, tuple) or len(pose_result) != 2:
        return
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0.0:
        return

    box_width, box_height, box_depth = box_size
    rotation_vector, translation_vector = pose_result
    half_width = box_width * 0.5
    half_height = box_height * 0.5
    half_depth = box_depth * 0.5
    box_points = np.array(
        [
            (-half_width, -half_height, -half_depth),
            (half_width, -half_height, -half_depth),
            (half_width, half_height, -half_depth),
            (-half_width, half_height, -half_depth),
            (-half_width, -half_height, half_depth),
            (half_width, -half_height, half_depth),
            (half_width, half_height, half_depth),
            (-half_width, half_height, half_depth),
        ],
        dtype=np.float32,
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    box_points_camera = (rotation_matrix @ box_points.T).T + translation_vector.reshape(1, 3)
    if not np.isfinite(box_points_camera).all() or np.any(box_points_camera[:, 2] <= 1e-6):
        return
    projected_box_points, _ = cv2.projectPoints(box_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients)
    projected_box_points = projected_box_points.reshape(-1, 2)
    if not np.isfinite(projected_box_points).all():
        return
    projected_box_points = np.rint(projected_box_points).astype(np.int32)
    overlay = np.copy(frame)

    box_faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    )
    visible_edges = set()
    for face in box_faces:
        p0 = box_points_camera[face[0]]
        p1 = box_points_camera[face[1]]
        p2 = box_points_camera[face[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        face_center = np.mean(box_points_camera[list(face)], axis=0)
        if np.dot(normal, face_center) < 0.0:
            for index in range(len(face)):
                start_index = face[index]
                end_index = face[(index + 1) % len(face)]
                visible_edges.add(tuple(sorted((start_index, end_index))))

    for start_index, end_index in sorted(visible_edges):
        start = tuple(projected_box_points[start_index])
        end = tuple(projected_box_points[end_index])
        cv2.line(overlay, start, end, (255, 255, 255), 3)

    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0.0, frame)


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
