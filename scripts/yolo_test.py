from dataclasses import dataclass
from pathlib import Path
import sys

import cv2

from src import yolo

WINDOW_NAME = "YOLO Test"


@dataclass
class YoloTestConfig:
    input_source: int | str | Path = 0
    model_path: Path = Path(r"C:\ComputerVision\_Datasets_\tb_dataManager\box_tracker_tracker\train\weights\best.engine")
    # model_path: Path = Path("models/best.engine")
    yolo_confidence: float = 0.7
    yolo_iou: float = 0.5
    yolo_input_size: int = 640
    camera_width: int = 640
    camera_height: int = 480


def main():
    config = YoloTestConfig()
    input_source = config.input_source
    video_capture_source = str(input_source) if isinstance(input_source, Path) else input_source

    detection_model = yolo.load_detection_model(
        config.model_path,
        config.yolo_confidence,
        config.yolo_iou,
        config.yolo_input_size,
    )
    print(detection_model.describe())

    cap = cv2.VideoCapture(video_capture_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

    if not cap.isOpened():
        print(f"Error: Could not open input source {input_source}.")
        sys.exit(1)

    capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {capture_width}x{capture_height}")

    while True:
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        if not ret:
            print("Error: Failed to read frame.")
            break

        detections = detection_model.predict(frame)
        yolo.draw_yolo_overlay(frame, detections)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKeyEx(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
