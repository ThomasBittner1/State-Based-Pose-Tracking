import json
import sys
from pathlib import Path

import cv2

WINDOW_NAME = "Image Annotation"
RECTANGLE_COLOR = (0, 255, 255)
PREVIEW_COLOR = (255, 255, 0)
CAPTURES_DIR = Path("captures")


class AnnotationTool:
    def __init__(self, capture_name):
        self.capture_name = capture_name
        self.image_path = CAPTURES_DIR / f"{capture_name}.png"
        self.json_path = CAPTURES_DIR / f"{capture_name}.json"
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise FileNotFoundError(f"Could not open image: {self.image_path}")

        self.rectangle_points = []
        self.load_annotations()

    def load_annotations(self):
        if not self.json_path.exists():
            self.save_annotations()
            return

        try:
            with self.json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            self.save_annotations()
            return

        raw_rectangle_points = data.get("rectangle_points", [])
        self.rectangle_points = [self._normalize_point(point) for point in raw_rectangle_points if self._normalize_point(point) is not None][:4]
        self.save_annotations()

    def save_annotations(self):
        payload = {
            "rectangle_points": [list(point) for point in self.rectangle_points],
        }
        with self.json_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @staticmethod
    def _normalize_point(point):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        return (int(point[0]), int(point[1]))

    def add_rectangle_point(self, x, y):
        if len(self.rectangle_points) >= 4:
            self.rectangle_points.pop(0)
        self.rectangle_points.append((int(x), int(y)))
        self.save_annotations()

    def pop_last_rectangle_point(self):
        if self.rectangle_points:
            self.rectangle_points.pop()
            self.save_annotations()

    def draw(self):
        preview = self.image.copy()

        for index, point in enumerate(self.rectangle_points, start=1):
            cv2.circle(preview, point, 6, RECTANGLE_COLOR, -1)
            cv2.putText(preview, f"R{index}", (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RECTANGLE_COLOR, 2, cv2.LINE_AA)
        for index in range(len(self.rectangle_points) - 1):
            cv2.line(preview, self.rectangle_points[index], self.rectangle_points[index + 1], PREVIEW_COLOR, 2)
        if len(self.rectangle_points) == 4:
            cv2.line(preview, self.rectangle_points[-1], self.rectangle_points[0], PREVIEW_COLOR, 2)

        instructions = [
            "LMB: add rectangle point",
            "U: undo rectangle point",
            "Q: quit",
        ]
        for index, line in enumerate(instructions):
            cv2.putText(preview, line, (20, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, preview)

    def on_mouse(self, event, x, y, _flags, _param):
        point = (int(x), int(y))

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self.add_rectangle_point(*point)


def get_capture_name():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python box_annotate_image.py <name>")
        sys.exit(1)
    capture_name = sys.argv[1].strip()
    json_path = CAPTURES_DIR / f"{capture_name}.json"
    if json_path.exists():
        response = input(f"{json_path} already exists. Overwrite? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)
    return capture_name


def main():
    CAPTURES_DIR.mkdir(exist_ok=True)
    capture_name = get_capture_name()
    tool = AnnotationTool(capture_name)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, tool.on_mouse)

    while True:
        tool.draw()
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("u"), ord("U")):
            tool.pop_last_rectangle_point()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
