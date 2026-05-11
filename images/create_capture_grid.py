import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURES_DIR = PROJECT_ROOT / "captures"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "captures_2x2.png"
DEFAULT_CAPTURE_NAMES = ("front", "right", "back", "left")

RECTANGLE_COLOR = (0, 255, 255)
LINE_COLOR = (255, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW_COLOR = (0, 0, 0)


def load_rectangle_points(json_path):
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    points = np.array(data.get("rectangle_points", []), dtype=np.int32)
    if points.shape != (4, 2):
        raise ValueError(f"{json_path} must contain exactly four rectangle_points")
    return points


def draw_text(image, text, origin, scale=0.8, color=TEXT_COLOR, thickness=2):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_SHADOW_COLOR, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_json_overlay(image, capture_name, points):
    cv2.polylines(image, [points], isClosed=True, color=LINE_COLOR, thickness=3, lineType=cv2.LINE_AA)

    for index, point in enumerate(points, start=1):
        point_tuple = (int(point[0]), int(point[1]))
        cv2.circle(image, point_tuple, 7, RECTANGLE_COLOR, -1, lineType=cv2.LINE_AA)
        draw_text(image, f"R{index}", (point_tuple[0] + 10, point_tuple[1] - 10), scale=0.55, color=RECTANGLE_COLOR, thickness=2)

    draw_text(image, capture_name, (20, 38), scale=1.0, thickness=2)


def load_annotated_capture(captures_dir, capture_name):
    image_path = captures_dir / f"{capture_name}.png"
    json_path = captures_dir / f"{capture_name}.json"

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Could not open annotations: {json_path}")

    points = load_rectangle_points(json_path)
    draw_json_overlay(image, capture_name, points)
    return image


def resize_to_tile(image, tile_size):
    tile_width, tile_height = tile_size
    return cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)


def make_grid(images):
    first_height, first_width = images[0].shape[:2]
    tile_size = (first_width, first_height)
    tiles = [resize_to_tile(image, tile_size) for image in images]

    top_row = np.hstack((tiles[0], tiles[1]))
    bottom_row = np.hstack((tiles[2], tiles[3]))
    return np.vstack((top_row, bottom_row))


def parse_args():
    parser = argparse.ArgumentParser(description="Create a 2x2 image of the annotated capture images.")
    parser.add_argument(
        "--captures-dir",
        type=Path,
        default=DEFAULT_CAPTURES_DIR,
        help=f"Directory containing capture .png and .json files. Default: {DEFAULT_CAPTURES_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output image path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "capture_names",
        nargs="*",
        default=DEFAULT_CAPTURE_NAMES,
        help="Exactly four capture basenames to place left-to-right, top-to-bottom.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    capture_names = tuple(args.capture_names)
    if len(capture_names) != 4:
        raise ValueError("Provide exactly four capture names for the 2x2 grid")

    captures_dir = args.captures_dir.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated_images = [load_annotated_capture(captures_dir, capture_name) for capture_name in capture_names]
    grid = make_grid(annotated_images)

    if not cv2.imwrite(str(output_path), grid):
        raise OSError(f"Could not write output image: {output_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
