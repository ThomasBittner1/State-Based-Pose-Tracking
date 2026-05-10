import cv2
import numpy as np


LABEL_PATCH_WIDTH = 320
LABEL_MASK_CACHE = {}


def _render_label_masks(label, face_width, face_height):
    aspect_ratio = round(float(face_height) / max(float(face_width), 1e-6), 4)
    cache_key = (label, aspect_ratio)
    if cache_key in LABEL_MASK_CACHE:
        return LABEL_MASK_CACHE[cache_key]

    patch_width = LABEL_PATCH_WIDTH
    patch_height = max(96, int(round(patch_width * face_height / max(face_width, 1e-6))))
    patch_height = min(patch_height, 480)
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_size, _ = cv2.getTextSize(label, font, 1.0, 1)
    font_scale = min(
        patch_width * 0.62 / max(base_size[0], 1),
        patch_height * 0.22 / max(base_size[1], 1),
    )
    thickness = max(1, int(round(font_scale * 2.0)))
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    origin = (
        max(0, (patch_width - text_size[0]) // 2),
        max(text_size[1], (patch_height + text_size[1]) // 2 - baseline),
    )

    fill_mask = np.zeros((patch_height, patch_width), dtype=np.uint8)
    outline_mask = np.zeros_like(fill_mask)
    cv2.putText(outline_mask, label, origin, font, font_scale, 255, thickness + 3, cv2.LINE_AA)
    cv2.putText(fill_mask, label, origin, font, font_scale, 255, thickness, cv2.LINE_AA)
    LABEL_MASK_CACHE[cache_key] = outline_mask, fill_mask
    return outline_mask, fill_mask


def _blend_mask(frame, mask, color):
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    frame[:] = (frame.astype(np.float32) * (1.0 - alpha) + color_array * alpha).astype(np.uint8)


def _draw_face_label(frame, label, destination_quad, face_width, face_height):
    if abs(cv2.contourArea(destination_quad.astype(np.float32))) < 64.0:
        return
    min_x = max(0, int(np.floor(np.min(destination_quad[:, 0]))) - 2)
    min_y = max(0, int(np.floor(np.min(destination_quad[:, 1]))) - 2)
    max_x = min(frame.shape[1], int(np.ceil(np.max(destination_quad[:, 0]))) + 3)
    max_y = min(frame.shape[0], int(np.ceil(np.max(destination_quad[:, 1]))) + 3)
    if max_x <= min_x or max_y <= min_y:
        return

    outline_mask, fill_mask = _render_label_masks(label, face_width, face_height)
    source_quad = np.array(
        [
            [0.0, 0.0],
            [outline_mask.shape[1] - 1.0, 0.0],
            [outline_mask.shape[1] - 1.0, outline_mask.shape[0] - 1.0],
            [0.0, outline_mask.shape[0] - 1.0],
        ],
        dtype=np.float32,
    )
    local_destination_quad = destination_quad.astype(np.float32) - np.array([min_x, min_y], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source_quad, local_destination_quad)
    roi_size = (max_x - min_x, max_y - min_y)
    warped_outline = cv2.warpPerspective(
        outline_mask,
        homography,
        roi_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_fill = cv2.warpPerspective(
        fill_mask,
        homography,
        roi_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    frame_roi = frame[min_y:max_y, min_x:max_x]
    _blend_mask(frame_roi, warped_outline, (0, 0, 0))
    _blend_mask(frame_roi, warped_fill, (255, 255, 255))


def _plane_corners_in_box_coordinates(reference):
    rotation_offset = np.asarray(reference.rotation_offset, dtype=np.float64)
    translation_offset = np.asarray(reference.translation_offset, dtype=np.float64).reshape(1, 3)
    plane_corners = np.asarray(reference.four_corner_points_3d, dtype=np.float64)
    return ((plane_corners - translation_offset) @ rotation_offset).astype(np.float32)


def _is_face_visible(face_points, face_points_camera):
    p0, p1, p2 = face_points[:3]
    normal_box = np.cross(p1 - p0, p2 - p0)
    face_center_box = np.mean(face_points, axis=0)
    p0_camera, p1_camera, p2_camera = face_points_camera[:3]
    normal = np.cross(p1_camera - p0_camera, p2_camera - p0_camera)
    if np.dot(normal_box, face_center_box) < 0.0:
        normal = -normal
    face_center = np.mean(face_points_camera, axis=0)
    return np.dot(normal, face_center) < 0.0


def draw_box_overlay(frame, camera_matrix, distortion_coefficients, pose_result, references, opacity=1.0, draw_labels=True):
    if not isinstance(pose_result, tuple) or len(pose_result) != 2:
        return
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0.0:
        return

    rotation_vector, translation_vector = pose_result
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    translation_vector = np.asarray(translation_vector, dtype=np.float64).reshape(1, 3)
    reference_faces = []
    for reference in references:
        if not all(hasattr(reference, attribute) for attribute in ("four_corner_points_3d", "rotation_offset", "translation_offset")):
            continue
        face_points = _plane_corners_in_box_coordinates(reference)
        face_points_camera = (rotation_matrix @ face_points.T).T + translation_vector
        if not np.isfinite(face_points_camera).all() or np.any(face_points_camera[:, 2] <= 1e-6):
            continue
        if not _is_face_visible(face_points, face_points_camera):
            continue
        projected_points, _ = cv2.projectPoints(face_points, rotation_vector, translation_vector.reshape(3), camera_matrix, distortion_coefficients)
        projected_points = projected_points.reshape(-1, 2)
        if not np.isfinite(projected_points).all():
            continue
        reference_faces.append((reference, face_points, projected_points))

    if not reference_faces:
        return
    overlay = np.copy(frame)

    for _reference, _face_points, projected_points in reference_faces:
        polygon = np.rint(projected_points).astype(np.int32)
        cv2.polylines(overlay, [polygon], isClosed=True, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    if draw_labels:
        for reference, face_points, projected_points in reference_faces:
            face_width = float(np.linalg.norm(face_points[1] - face_points[0]))
            face_height = float(np.linalg.norm(face_points[3] - face_points[0]))
            _draw_face_label(overlay, reference.name, projected_points, face_width, face_height)

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
