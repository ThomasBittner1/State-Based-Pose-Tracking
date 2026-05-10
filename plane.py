from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json

import cv2
import numpy as np

import drawing
import geometry


SUPPORTED_ARUCO_DICTIONARIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "7x7_50": cv2.aruco.DICT_7X7_50,
}

FeatureDetectorName = Literal["ORB", "AKAZE"]
SUPPORTED_FEATURE_DETECTORS = {"ORB", "AKAZE"}


class FeatureDetector:
    ORB: FeatureDetectorName = "ORB"
    AKAZE: FeatureDetectorName = "AKAZE"


@dataclass
class PlaneTrackingConfig:
    feature_detector: FeatureDetectorName = "ORB"
    bruteforce_matcher: bool = True
    min_match_count: int = 8
    ransac_threshold: float = 4.0
    flow_max_error: float = 20.0
    flow_window_size: tuple[int, int] = (21, 21)
    flow_max_level: int = 3
    draw_homography_outline: bool = False
    straighten_z_on_front: bool = True
    straight_rotation_start_angle_degrees: float = 5.0
    straight_rotation_end_angle_degrees: float = 10.0
    yolo_bounds_history_size: int = 4


@dataclass
class Aruco:
    id: int
    dictionary_name: str
    points_3d: np.ndarray
    plane: "Plane"


@dataclass
class ArucoRegistry:
    dictionaries: dict[str, int] = field(default_factory=lambda: dict(SUPPORTED_ARUCO_DICTIONARIES))
    detectors: dict[str, cv2.aruco.ArucoDetector] = field(init=False)
    used_dictionary_names: set[str] = field(default_factory=set)
    markers_by_dictionary: defaultdict = field(default_factory=lambda: defaultdict(dict))

    def __post_init__(self):
        self.detectors = {
            dictionary_name: cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(dictionary_id),
                cv2.aruco.DetectorParameters(),
            )
            for dictionary_name, dictionary_id in self.dictionaries.items()
        }

    def detect_markers(self, dictionary_name, image):
        return self.detectors[dictionary_name].detectMarkers(image)

    def register_marker(self, aruco):
        markers_for_dictionary = self.markers_by_dictionary[aruco.dictionary_name]
        if aruco.id in markers_for_dictionary:
            existing = markers_for_dictionary[aruco.id]
            raise ValueError(
                f"Duplicate Aruco Marker found: {aruco.dictionary_name}, {aruco.id} "
                f"({aruco.plane.name} -> {existing.plane.name})"
            )
        markers_for_dictionary[aruco.id] = aruco
        self.used_dictionary_names.add(aruco.dictionary_name)

    def get_marker(self, dictionary_name, marker_id):
        return self.markers_by_dictionary[dictionary_name].get(int(marker_id))


class PoseHistory:
    def __init__(self, min_similarity=0.7):
        self.min_similarity = float(min_similarity)
        self.rotation_matrices = []

    def accepts(self, rotation_matrix):
        if not self.rotation_matrices:
            self.rotation_matrices = [rotation_matrix]
            return True

        for previous_rotation_matrix in self.rotation_matrices:
            rotation_similarity = geometry.rotation_matrix_similarity(previous_rotation_matrix, rotation_matrix)
            if rotation_similarity > self.min_similarity:
                self.rotation_matrices = [rotation_matrix]
                return True

        self.rotation_matrices.append(rotation_matrix)
        return False


class Plane:
    def __init__(
        self,
        name,
        json_path,
        aruco_registry,
        pose_history,
        config=None,
        world_size=(1.0, 1.0),
        display_color_multiplier=1.0,
    ):
        self.name = name
        self.aruco_registry = aruco_registry
        self.pose_history = pose_history
        self.config = config or PlaneTrackingConfig()
        self.previous_tracking = {"frame_gray": None, "points_by_query": {}}
        self.good_matches = []
        self.key_points_on_full_frame = []
        self.homography = None
        self.inlier_mask = []
        self.optical_flow_query_indices = set()
        self.inlier_matches = []
        self.pose_result = None
        self.world_size = world_size
        self.display_color_multiplier = float(np.clip(display_color_multiplier, 0.0, 1.0))
        self.crop_reference_image(json_path)

    def get_scaled_reference_size(self, reference_column_width):
        self.reference_height, self.reference_width = self.warped_reference_img.shape[:2]
        scale = reference_column_width / max(1, self.reference_width)
        scaled_height = max(1, int(round(self.reference_height * scale)))
        return reference_column_width, scaled_height, scale

    def get_display_color(self, color):
        gray = sum(color) / 3.0
        return tuple(int(round(gray + (channel - gray) * self.display_color_multiplier)) for channel in color)

    def crop_reference_image(self, image_or_json_path):
        input_path = Path(image_or_json_path)
        json_path = input_path.with_suffix(".json")
        image_path = json_path.with_suffix(".png")

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        rectangle_points = np.array(data.get("rectangle_points", []), dtype=np.float32)
        if rectangle_points.shape != (4, 2):
            raise ValueError("rectangle_points must contain exactly 4 points")

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not open image: {image_path}")

        ordered = geometry.order_rectangle_points(rectangle_points)
        top_left, top_right, bottom_right, bottom_left = ordered

        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)

        target_width = max(1, int(round(max(width_top, width_bottom))))
        target_height = max(1, int(round(max(height_left, height_right))))
        self.ratio = target_width / target_height

        destination = np.array(
            [
                [0.0, 0.0],
                [target_width - 1.0, 0.0],
                [target_width - 1.0, target_height - 1.0],
                [0.0, target_height - 1.0],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(ordered, destination)
        self.warped_reference_img = cv2.warpPerspective(image, transform, (target_width, target_height))
        self.reference_height, self.reference_width = self.warped_reference_img.shape[:2]
        self.reference_center_x = (self.reference_width - 1) * 0.5
        self.reference_center_y = (self.reference_height - 1) * 0.5


    def compute_feature_correspondences(self, plane_size, rotation_offset=None, translation_offset=(0.0, 0.0, 0.0)):
        if rotation_offset is None:
            rotation_offset = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.rotation_offset = np.array(rotation_offset, dtype=np.float64)
        self.translation_offset = np.array(translation_offset, dtype=np.float64)

        self.four_corner_points_3d = np.array(
            [
                (-plane_size[0] * 0.5, -plane_size[1] * 0.5, 0.0),
                (plane_size[0] * 0.5, -plane_size[1] * 0.5, 0.0),
                (plane_size[0] * 0.5, plane_size[1] * 0.5, 0.0),
                (-plane_size[0] * 0.5, plane_size[1] * 0.5, 0.0),
            ],
            dtype=np.float32,
        )
        width_scale = plane_size[0] / max(1.0, float(self.reference_width - 1))
        height_scale = plane_size[1] / max(1.0, float(self.reference_height - 1))

        mirrored_warped_reference_img = cv2.flip(self.warped_reference_img, 1)
        for dictionary_name in self.aruco_registry.dictionaries:
            corners, ids, _rejected = self.aruco_registry.detect_markers(dictionary_name, mirrored_warped_reference_img)
            if ids is not None and len(ids):
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    points_3d = np.array(
                        [
                            [
                                width_scale - (corner[0] - self.reference_center_x) * width_scale,
                                (corner[1] - self.reference_center_y) * height_scale,
                                0.0,
                            ]
                            for corner in marker_corners[0]
                        ],
                        dtype=np.float32,
                    )
                    aruco = Aruco(int(marker_id), dictionary_name, points_3d, self)
                    self.aruco_registry.register_marker(aruco)
                    print(f"found ArUco for {self.name}: {dictionary_name}, {marker_id}")

        reference_img_gray = cv2.cvtColor(self.warped_reference_img, cv2.COLOR_BGR2GRAY)
        detector_name = self.config.feature_detector.upper()

        if detector_name == "AKAZE":
            self.detector = cv2.AKAZE_create()
        elif detector_name == "ORB":
            self.detector = cv2.ORB_create(1500, nlevels=8)
        else:
            supported_detectors = ", ".join(sorted(SUPPORTED_FEATURE_DETECTORS))
            raise ValueError(
                f"Unsupported feature detector: {self.config.feature_detector}. "
                f"Choose one of: {supported_detectors}"
            )

        self.reference_keypoints, self.reference_descriptors = self.detector.detectAndCompute(reference_img_gray, None)
        if self.reference_descriptors is None or len(self.reference_keypoints) == 0:
            raise RuntimeError("Could not extract features from reference image")
        self.reference_points_3d = np.array(
            [
                [(keypoint.pt[0] - self.reference_center_x) * width_scale, (keypoint.pt[1] - self.reference_center_y) * height_scale, 0.0]
                for keypoint in self.reference_keypoints
            ],
            dtype=np.float32,
        )
        if self.config.bruteforce_matcher:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            index_params = dict(
                algorithm=6,
                table_number=12,
                key_size=20,
                multi_probe_level=2,
            )
            search_params = dict(checks=32)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def reset_tracking_result(self):
        self.good_matches = []
        self.key_points_on_full_frame = []
        self.homography = None
        self.inlier_mask = []
        self.optical_flow_query_indices = set()
        self.inlier_matches = []
        self.pose_result = None

    def find_matches(self, frame_gray, combined_yolo_bounds, concat_points=False):
        if combined_yolo_bounds is None:
            self.reset_tracking_result()
            return 0.0

        left, top, right, bottom = combined_yolo_bounds
        cropped_frame = frame_gray[top:bottom, left:right]
        keypoints, descriptors = self.detector.detectAndCompute(cropped_frame, None)
        self.key_points_on_full_frame = []
        for keypoint in keypoints:
            self.key_points_on_full_frame.append(
                cv2.KeyPoint(
                    x=float(keypoint.pt[0] + left),
                    y=float(keypoint.pt[1] + top),
                    size=keypoint.size,
                    angle=keypoint.angle,
                    response=keypoint.response,
                    octave=keypoint.octave,
                    class_id=keypoint.class_id,
                )
            )

        if descriptors is None or len(keypoints) == 0:
            self.previous_tracking["frame_gray"] = frame_gray.copy()
            self.previous_tracking["points_by_query"] = {}
            self.good_matches = []
            self.optical_flow_query_indices = set()
            self.homography = None
            self.inlier_mask = []
            self.inlier_matches = []
            self.pose_result = None
            return 0.0

        knn_matches = self.matcher.knnMatch(self.reference_descriptors, descriptors, k=2)
        self.good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                self.good_matches.append(first)

        if self.previous_tracking["frame_gray"] is None:
            self.optical_flow_query_indices = set()
        elif self.previous_tracking["frame_gray"].shape != frame_gray.shape:
            self.previous_tracking["frame_gray"] = None
            self.previous_tracking["points_by_query"] = {}
            self.optical_flow_query_indices = set()
        elif not self.previous_tracking["points_by_query"]:
            self.optical_flow_query_indices = set()
        else:
            matched_query_indices = {match.queryIdx for match in self.good_matches}
            missing_query_indices = [
                query_index for query_index in self.previous_tracking["points_by_query"] if query_index not in matched_query_indices
            ]
            if not missing_query_indices:
                self.optical_flow_query_indices = set()
            else:
                previous_points = np.float32(
                    [self.previous_tracking["points_by_query"][query_index] for query_index in missing_query_indices]
                ).reshape(-1, 1, 2)
                tracked_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.previous_tracking["frame_gray"],
                    frame_gray,
                    previous_points,
                    None,
                    winSize=self.config.flow_window_size,
                    maxLevel=self.config.flow_max_level,
                )
                self.optical_flow_query_indices = set()
                if tracked_points is not None and status is not None:
                    frame_height, frame_width = frame_gray.shape[:2]
                    recovered_matches = list(self.good_matches)
                    recovered_keypoints = list(self.key_points_on_full_frame)
                    for query_index, point, keep, point_error in zip(
                        missing_query_indices,
                        tracked_points.reshape(-1, 2),
                        status.ravel(),
                        error.ravel(),
                    ):
                        if not keep or point_error > self.config.flow_max_error:
                            continue
                        x, y = point
                        if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
                            continue
                        if x < left or x >= right or y < top or y >= bottom:
                            continue
                        recovered_keypoints.append(cv2.KeyPoint(float(x), float(y), 8.0))
                        recovered_matches.append(
                            cv2.DMatch(_queryIdx=query_index, _trainIdx=len(recovered_keypoints) - 1, _distance=float(point_error))
                        )
                        self.optical_flow_query_indices.add(query_index)
                    self.good_matches = recovered_matches
                    self.key_points_on_full_frame = recovered_keypoints

        self.previous_tracking["frame_gray"] = frame_gray.copy()
        self.previous_tracking["points_by_query"] = {
            match.queryIdx: np.array(self.key_points_on_full_frame[match.trainIdx].pt, dtype=np.float32)
            for match in self.good_matches
        }

        if len(self.good_matches) < self.config.min_match_count:
            self.homography, self.inlier_mask = None, []
        else:
            reference_points = np.float32([self.reference_keypoints[match.queryIdx].pt for match in self.good_matches]).reshape(-1, 1, 2)
            frame_points = np.float32([self.key_points_on_full_frame[match.trainIdx].pt for match in self.good_matches]).reshape(-1, 1, 2)
            self.homography, mask = cv2.findHomography(reference_points, frame_points, cv2.RANSAC, self.config.ransac_threshold)
            self.inlier_mask = [] if mask is None else mask.ravel().astype(bool).tolist()
            inlier_count = sum(self.inlier_mask)
            if inlier_count < self.config.min_match_count:
                self.homography, self.inlier_mask = None, []

        self.inlier_matches = [match for match, keep in zip(self.good_matches, self.inlier_mask) if keep] if self.inlier_mask else []

        homography_confidence = geometry.rate_homography(self.homography)

        if self.inlier_matches and homography_confidence > 0.9:
            points_3d = np.array([self.reference_points_3d[match.queryIdx] for match in self.inlier_matches], dtype=np.float32)
            points_2d = np.array([self.key_points_on_full_frame[match.trainIdx].pt for match in self.inlier_matches], dtype=np.float32)
            if concat_points:
                self.points_3d = np.concatenate([self.points_3d, points_3d])
                self.points_2d = np.concatenate([self.points_2d, points_2d])
            else:
                self.points_3d = points_3d
                self.points_2d = points_2d
        else:
            if not concat_points:
                self.points_3d = np.zeros(0, dtype=np.float32)
                self.points_2d = np.zeros(0, dtype=np.float32)

        return homography_confidence

    def estimate_pose_from_matches(self, camera_matrix, distortion_coefficients, frame_debug=None):
        success, rotation_vectors, translation_vectors, _ = cv2.solvePnPGeneric(
            self.points_3d,
            self.points_2d,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if success and frame_debug is not None:
            debug_colors = ((255, 255, 255), (255, 255, 255))
            for pose_index, (rotation_vector, translation_vector) in enumerate(zip(rotation_vectors[:2], translation_vectors[:2])):
                drawing.draw_pose_axes_overlay(
                    frame_debug,
                    camera_matrix,
                    distortion_coefficients,
                    (rotation_vector, np.array(translation_vector.reshape(3), dtype=np.float64)),
                    debug_colors[pose_index % len(debug_colors)],
                    f"IPPE {pose_index}",
                )

        if not success:
            self.pose_result = None
            return False

        used_index = 0
        translation_vector = np.array(translation_vectors[used_index].reshape(3), dtype=np.float64)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vectors[used_index])

        if self.config.straighten_z_on_front:
            z_axis = rotation_matrix[:, 2]
            z_axis = z_axis / np.linalg.norm(z_axis)
            horizontal_angle_degrees = float(np.degrees(np.arctan2(abs(z_axis[0]), abs(z_axis[2]))))
            vertical_angle_degrees = float(np.degrees(np.arctan2(abs(z_axis[1]), abs(z_axis[2]))))
            horizontal_blend = 1.0 - np.clip(
                (horizontal_angle_degrees - self.config.straight_rotation_start_angle_degrees * 2)
                / (self.config.straight_rotation_end_angle_degrees * 2 - self.config.straight_rotation_start_angle_degrees * 2),
                0.0,
                1.0,
            )
            vertical_blend = 1.0 - np.clip(
                (vertical_angle_degrees - self.config.straight_rotation_start_angle_degrees)
                / (self.config.straight_rotation_end_angle_degrees - self.config.straight_rotation_start_angle_degrees),
                0.0,
                1.0,
            )
            target_z_axis = -translation_vector / np.linalg.norm(translation_vector)
            straight_z_axis = np.array(
                [
                    z_axis[0] * (1.0 - horizontal_blend) + target_z_axis[0] * horizontal_blend,
                    z_axis[1] * (1.0 - vertical_blend) + target_z_axis[1] * vertical_blend,
                    z_axis[2],
                ],
                dtype=np.float64,
            )
            straight_z_axis /= np.linalg.norm(straight_z_axis)
            old_x_axis = rotation_matrix[:, 0]
            old_y_axis = rotation_matrix[:, 1]
            straight_y_axis = old_y_axis - np.dot(old_y_axis, straight_z_axis) * straight_z_axis
            if np.linalg.norm(straight_y_axis) < 1e-8:
                straight_x_axis = old_x_axis - np.dot(old_x_axis, straight_z_axis) * straight_z_axis
                straight_x_axis /= np.linalg.norm(straight_x_axis)
                straight_y_axis = np.cross(straight_z_axis, straight_x_axis)
                straight_y_axis /= np.linalg.norm(straight_y_axis)
            else:
                straight_y_axis /= np.linalg.norm(straight_y_axis)
                straight_x_axis = np.cross(straight_y_axis, straight_z_axis)
                straight_x_axis /= np.linalg.norm(straight_x_axis)
            straight_y_axis = np.cross(straight_z_axis, straight_x_axis)
            straight_y_axis /= np.linalg.norm(straight_y_axis)
            rotation_matrix = np.column_stack((straight_x_axis, straight_y_axis, straight_z_axis))

        translation_vector = translation_vector + rotation_matrix @ self.translation_offset
        rotation_matrix = rotation_matrix @ np.array(self.rotation_offset, dtype=np.float64)
        if frame_debug is not None:
            debug_rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
            debug_axis_points = np.array([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)], dtype=np.float32)
            debug_axis_points_2d, _ = cv2.projectPoints(
                debug_axis_points,
                debug_rotation_vector,
                translation_vector,
                camera_matrix,
                distortion_coefficients,
            )
            debug_axis_points_2d = np.rint(debug_axis_points_2d.reshape(-1, 2)).astype(np.int32)
            debug_axis_origin = tuple(debug_axis_points_2d[0])
            cv2.line(frame_debug, debug_axis_origin, tuple(debug_axis_points_2d[1]), (0, 0, 125), 3)
            cv2.line(frame_debug, debug_axis_origin, tuple(debug_axis_points_2d[2]), (0, 125, 0), 3)
            cv2.line(frame_debug, debug_axis_origin, tuple(debug_axis_points_2d[3]), (125, 0, 0), 3)

        if not self.pose_history.accepts(rotation_matrix):
            return False

        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        self.pose_result = rotation_vector, translation_vector
        return True

    def draw(self, frame, camera_matrix, distortion_coefficients):
        if isinstance(self.pose_result, tuple) and len(self.pose_result) == 2:
            rotation_vector, translation_vector = self.pose_result

            axis_points = np.array([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)], dtype=np.float32)
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            axis_points_camera = (rotation_matrix @ axis_points.T).T + translation_vector.reshape(1, 3)
            if not np.isfinite(axis_points_camera).all() or np.any(axis_points_camera[:, 2] <= 1e-6):
                return
            projected_points, _ = cv2.projectPoints(axis_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients)
            projected_points = projected_points.reshape(-1, 2)
            if np.isfinite(projected_points).all():
                projected_points = np.rint(projected_points).astype(np.int32)
                origin = (int(projected_points[0, 0]), int(projected_points[0, 1]))
                x_axis = (int(projected_points[1, 0]), int(projected_points[1, 1]))
                y_axis = (int(projected_points[2, 0]), int(projected_points[2, 1]))
                z_axis = (int(projected_points[3, 0]), int(projected_points[3, 1]))
                cv2.line(frame, origin, x_axis, self.get_display_color((0, 0, 255)), 3)
                cv2.line(frame, origin, y_axis, self.get_display_color((0, 255, 0)), 3)
                cv2.line(frame, origin, z_axis, self.get_display_color((255, 0, 0)), 3)

        if self.config.draw_homography_outline and self.homography is not None:
            # self.reference_height, self.reference_width = self.warped_reference_img.shape[:2]
            reference_corners = np.array(
                [[0.0, 0.0], [self.reference_width - 1.0, 0.0], [self.reference_width - 1.0, self.reference_height - 1.0], [0.0, self.reference_height - 1.0]],
                dtype=np.float32,
            )
            tracked_corners = cv2.perspectiveTransform(reference_corners.reshape(-1, 1, 2), self.homography).reshape(-1, 2)
            if np.isfinite(tracked_corners).all():
                tracked_corners = np.rint(tracked_corners).astype(int)
                for index in range(4):
                    start = tuple(tracked_corners[index])
                    end = tuple(tracked_corners[(index + 1) % 4])
                    cv2.line(frame, start, end, self.get_display_color((0, 255, 255)), 3)

    def draw_lines_to_reference(self, frame_with_references, reference_column_width, reference_top, homography_confidence):
        scaled_width, scaled_height, scale = self.get_scaled_reference_size(reference_column_width)
        stretched_reference = cv2.resize(self.warped_reference_img, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        frame_with_references[reference_top : reference_top + scaled_height, :scaled_width] = stretched_reference
        confidence_text = f"conf {homography_confidence:.2f}"
        cv2.putText(frame_with_references, confidence_text, (10, reference_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_with_references, confidence_text, (10, reference_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        matched_reference_indices = {match.queryIdx for match in self.good_matches}
        for index, keypoint in enumerate(self.reference_keypoints):
            if index not in matched_reference_indices:
                continue
            point = (
                int(round(keypoint.pt[0] * scale)),
                reference_top + int(round(keypoint.pt[1] * scale)),
            )
            color = (255, 0, 0) if index in self.optical_flow_query_indices else (0, 255, 0)
            color = self.get_display_color(color)
            radius = max(2, int(round(keypoint.size * 0.5)))
            cv2.circle(frame_with_references, point, radius, color, 1)
            cv2.circle(frame_with_references, point, 1, color, -1)

        for index, match in enumerate(self.good_matches):
            reference_point = (
                int(round(self.reference_keypoints[match.queryIdx].pt[0] * scale)),
                reference_top + int(round(self.reference_keypoints[match.queryIdx].pt[1] * scale)),
            )
            frame_point = tuple(np.rint(self.key_points_on_full_frame[match.trainIdx].pt).astype(int))
            frame_point_on_canvas = (frame_point[0] + reference_column_width, frame_point[1])
            is_inlier = self.inlier_mask[index] if index < len(self.inlier_mask) else False
            color = (255, 0, 0) if match.queryIdx in self.optical_flow_query_indices else (0, 255, 0)
            color = self.get_display_color(color)
            thickness = 4 if is_inlier else 1
            cv2.line(frame_with_references, reference_point, frame_point_on_canvas, color, thickness)
            cv2.circle(frame_with_references, frame_point_on_canvas, 4, color, -1)
