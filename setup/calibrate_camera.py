from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import camera


CAMERA_INDEX = 0
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 1.0
MIN_CAPTURES = 12
OUTPUT_PATH = PROJECT_ROOT / "calibration" / "camera.json"
PREVIEW_WINDOW = "Camera Calibration"
SPACE_KEY = 32


def create_object_points():
    object_points = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    object_points[:, :2] = np.indices(CHESSBOARD_SIZE).T.reshape(-1, 2)
    object_points *= SQUARE_SIZE
    return object_points


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        return

    termination = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    template_object_points = create_object_points()
    object_points = []
    image_points = []
    image_size = None

    print("Show a chessboard to the camera.")
    print("Press SPACE to capture a detected board.")
    print("Press C to calibrate once you have enough captures.")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        image_size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE)

        preview = frame.copy()
        message = f"captures: {len(image_points)}/{MIN_CAPTURES} | SPACE: capture | SPACE: calibrate | Q: quit"

        if found:
            refined_corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                termination,
            )
            cv2.drawChessboardCorners(preview, CHESSBOARD_SIZE, refined_corners, found)
            message += " | board detected"
        else:
            refined_corners = None

        cv2.putText(
            preview,
            message,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(PREVIEW_WINDOW, preview)

        key = cv2.waitKey(1) & 0xFF

        if key == SPACE_KEY:
            if refined_corners is None:
                print("No chessboard detected in the current frame.")
                continue

            object_points.append(template_object_points.copy())
            image_points.append(refined_corners)
            print(f"Captured frame {len(image_points)}.")

        if key in (ord("c"), ord("C")):
            if len(image_points) < MIN_CAPTURES:
                print(f"Need at least {MIN_CAPTURES} captures before calibration.")
                continue

            rms_error, camera_matrix, distortion_coefficients, _rvecs, _tvecs = cv2.calibrateCamera(
                object_points,
                image_points,
                image_size,
                None,
                None,
            )
            camera.save_camera_calibration(
                OUTPUT_PATH,
                camera_matrix,
                distortion_coefficients,
                image_size,
                rms_error,
                chessboard_size=CHESSBOARD_SIZE,
                square_size=SQUARE_SIZE,
            )
            print(f"Saved calibration to {OUTPUT_PATH}")
            print(f"RMS reprojection error: {rms_error:.4f}")
            print("Camera matrix:")
            print(camera_matrix)
            print("Distortion coefficients:")
            print(distortion_coefficients.ravel())
            break

        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
