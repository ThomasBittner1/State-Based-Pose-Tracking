import sys
import time
from pathlib import Path
import numpy as np
import cv2


CAMERA_INDEX = 0
WINDOW_NAME = "Record Image"
OUTPUT_DIR = Path("captures")
ARUCO_DICTIONARIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "7x7_50": cv2.aruco.DICT_7X7_50,
}


def draw_detected_arucos(source_frame, preview, mirrored=False):
    preview_width = preview.shape[1]
    for dictionary_name in ARUCO_DICTIONARIES:
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _rejected = detector.detectMarkers(source_frame)
        if ids is None or len(ids) == 0:
            continue
        preview_corners = []
        for marker_corners in corners:
            preview_marker_corners = marker_corners.copy()
            if mirrored:
                preview_marker_corners[0, :, 0] = (preview_width - 1) - preview_marker_corners[0, :, 0]
            preview_corners.append(preview_marker_corners)
        cv2.aruco.drawDetectedMarkers(preview, preview_corners, ids)
        for marker_corners, marker_id in zip(preview_corners, ids.flatten()):
            center = marker_corners[0].mean(axis=0)
            text_origin = (int(center[0] + 8), int(center[1] - 8))
            cv2.putText(preview, f"{dictionary_name}:{int(marker_id)}", text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, f"{dictionary_name}:{int(marker_id)}", text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def get_output_path():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python box_record_image.py <name>")
        sys.exit(1)

    capture_name = sys.argv[1].strip()
    output_path = OUTPUT_DIR / f"{capture_name}.png"

    return output_path


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = get_output_path()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print ('width: ', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print ('height: ', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        sys.exit(1)

    while True:
        loop_start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        preview = cv2.flip(frame, 1)
        draw_detected_arucos(frame, preview, mirrored=True)
        cv2.putText(preview, "SPACE: save image | Q: quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        elapsed_ms = (time.perf_counter() - loop_start_time) * 1000.0
        elapsed_text = f"{elapsed_ms:.1f} ms"
        (text_width, text_height), _ = cv2.getTextSize(elapsed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_origin = (preview.shape[1] - text_width - 20, 20 + text_height)
        cv2.putText(preview, elapsed_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, elapsed_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key == ord(" "):
            save_img = cv2.flip(frame, 1)
            cv2.imwrite(str(output_path), save_img)
            print(f"Saved {output_path}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
