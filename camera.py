from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np


@dataclass
class Calibration:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int]
    reprojection_error: float | None = None


def create_fallback_camera_matrix(frame_shape):
    frame_height, frame_width = frame_shape[:2]
    focal_length = float(max(frame_width, frame_height))
    return np.array(
        [
            [focal_length, 0.0, frame_width / 2.0],
            [0.0, focal_length, frame_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def create_fallback_calibration(frame_shape):
    frame_height, frame_width = frame_shape[:2]
    return Calibration(
        camera_matrix=create_fallback_camera_matrix(frame_shape),
        distortion_coefficients=np.zeros((5, 1), dtype=np.float32),
        image_size=(frame_width, frame_height),
    )


def load_calibration(path):
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None

    with calibration_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return Calibration(
        camera_matrix=np.array(payload["camera_matrix"], dtype=np.float32),
        distortion_coefficients=np.array(payload["distortion_coefficients"], dtype=np.float32).reshape(-1, 1),
        image_size=tuple(payload["image_size"]),
        reprojection_error=payload.get("reprojection_error"),
    )


def load_or_create_fallback(path, frame_shape):
    calibration = load_calibration(path)
    if calibration is not None:
        return calibration
    return create_fallback_calibration(frame_shape)


def save_camera_calibration(
    path,
    camera_matrix,
    distortion_coefficients,
    image_size,
    reprojection_error,
    chessboard_size=None,
    square_size=None,
):
    payload = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "camera_matrix": np.asarray(camera_matrix, dtype=float).tolist(),
        "distortion_coefficients": np.asarray(distortion_coefficients, dtype=float).reshape(-1).tolist(),
        "reprojection_error": float(reprojection_error),
    }
    if chessboard_size is not None:
        payload["chessboard_size"] = list(chessboard_size)
    if square_size is not None:
        payload["square_size"] = float(square_size)

    calibration_path = Path(path)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with calibration_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def undistort_frame(frame, calibration):
    return cv2.undistort(frame, calibration.camera_matrix, calibration.distortion_coefficients)
