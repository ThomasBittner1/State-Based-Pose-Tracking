from collections import deque, defaultdict
from pathlib import Path
import json
import sys
import time
import cv2
import numpy as np


import drawing_utils
import geometry_utils
import yolo_utils


FEATURE_DETECTOR = "ORB" # "AKAZE" or "ORB"
WINDOW_NAME = f"Box Track {FEATURE_DETECTOR}"
MODEL_PATH = Path(r"models/best.engine")
# MODEL_PATH = Path(r"runs\train9\weights\best.pt")
YOLO_CONFIDENCE = 0.1
YOLO_IOU = 0.1
ONNX_INPUT_SIZE = 640

BRUTEFORCE_MATCHER = True

NUM_CHECK_LAST_YOLO_BOX = 4
MIN_MATCH_COUNT = 8
RANSAC_THRESHOLD = 4.0
FLOW_MAX_ERROR = 20.0
FLOW_WINDOW_SIZE = (21, 21)
FLOW_MAX_LEVEL = 3
DRAW_HOMOGRAPHY_OUTLINE = False
STRAIGHT_ROTATION_START_ANGLE_DEGREES = 5.0
STRAIGHT_ROTATION_END_ANGLE_DEGREES = 10.0
STRAIGHTEN_Z_ON_FRONT = True
BOX_SIZE = (8.5, 14.0, 7.4) # width, height, depth
BOX_WIDTH, BOX_HEIGHT, BOX_DEPTH = BOX_SIZE

RECORD_WHEN_CAMERA = False
MOVIE_FILE_PATH = 'recorded_001.mp4'
CAMERA_INDEX_OR_FILE = 0 #'recorded_001.mp4'
START_FRAME = 60
START_PAUSED = False
LAST_POSES_CHECK = []
DEFAULT_FPS = 30.0
OUTPUT_FRAME_WIDTH = 1440
TIMING_AVERAGE_WINDOW = 10

SKIP_ALL = False
ENABLE_POSE_KALMAN = True


def create_fallback_camera_matrix(frame_shape):
    frame_height, frame_width = frame_shape[:2]
    focal_length = float(max(frame_width, frame_height))
    return np.array([[focal_length, 0.0, frame_width / 2.0], [0.0, focal_length, frame_height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

ARUCO_DICTIONARIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "7x7_50": cv2.aruco.DICT_7X7_50,
}


ARUCO_DETECTORS = {dictionary_name:cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name]),
                                                           cv2.aruco.DetectorParameters()) for dictionary_name in ARUCO_DICTIONARIES.keys()}
USED_ARUCO_DICTIONARIES = set()
ARUCO_PER_ID = defaultdict(lambda: defaultdict(list))


def draw_frame_number(frame, frame_rate=None, frame_number=None):
    frame_text = ""
    if frame_number:
        frame_text += f"frame {frame_number}"
    if frame_rate is not None and np.isfinite(frame_rate):
        frame_text += f" | {frame_rate:.1f} fps"
    (text_width, text_height), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_origin = (frame.shape[1] - text_width - 20, 20 + text_height)
    cv2.putText(frame, frame_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, frame_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)




class Aruco(object):
    def __init__(self, id, dictionary_name, points_3d, plane):
        self.id = id
        self.dictionary_name = dictionary_name
        self.points_3d = points_3d
        self.plane = plane


class Plane(object):
    def __init__(self, name, json_path, rotation_offset=[[1,0,0],[0,1,0],[0,0,1]], translation_offset=(0.0, 0.0, 0.0), world_size=(1.0, 1.0), display_color_multiplier=1.0):
        self.name = name
        self.previous_tracking = {"frame_gray": None, "points_by_query": {}}
        self.good_matches = []
        self.key_points_on_full_frame = []
        self.homography = None
        self.inlier_mask = []
        self.optical_flow_query_indices = set()
        self.inlier_matches = []
        self.pose_result = None
        self.world_size = world_size
        self.rotation_offset = np.array(rotation_offset, dtype=np.float64)
        self.translation_offset = np.array(translation_offset, dtype=np.float64)
        self.display_color_multiplier = float(np.clip(display_color_multiplier, 0.0, 1.0))
        self.warp_and_update_reference(json_path)

    def get_scaled_reference_size(self, reference_column_width):
        reference_height, reference_width = self.warped_reference_img.shape[:2]
        scale = reference_column_width / max(1, reference_width)
        scaled_height = max(1, int(round(reference_height * scale)))
        return reference_column_width, scaled_height, scale


    def get_display_color(self, color):
        gray = sum(color) / 3.0
        return tuple(int(round(gray + (channel - gray) * self.display_color_multiplier)) for channel in color)


    def warp_and_update_reference(self, image_or_json_path):
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

        ordered = geometry_utils.order_rectangle_points(rectangle_points)
        top_left, top_right, bottom_right, bottom_left = ordered

        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)


        target_width = max(1, int(round(max(width_top, width_bottom))))
        target_height = max(1, int(round(max(height_left, height_right))))
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

        self.four_corner_points_3d = np.array(
            [
                (-self.world_size[0] * 0.5, -self.world_size[1] * 0.5, 0.0),
                (self.world_size[0] * 0.5, -self.world_size[1] * 0.5, 0.0),
                (self.world_size[0] * 0.5, self.world_size[1] * 0.5, 0.0),
                (-self.world_size[0] * 0.5, self.world_size[1] * 0.5, 0.0),
            ],
            dtype=np.float32,
        )

        # Aruco
        #
        reference_height, reference_width = self.warped_reference_img.shape[:2]
        width_scale = self.world_size[0] / max(1.0, float(reference_width - 1))
        height_scale = self.world_size[1] / max(1.0, float(reference_height - 1))
        center_x = (reference_width - 1) * 0.5
        center_y = (reference_height - 1) * 0.5

        mirrored_warped_reference_img = cv2.flip(self.warped_reference_img, 1)
        global ARUCO_PER_ID
        for dictionary_name in ARUCO_DICTIONARIES:
            corners, ids, _rejected = ARUCO_DETECTORS[dictionary_name].detectMarkers(mirrored_warped_reference_img)
            if ids is not None and len(ids):
                global USED_ARUCO_DICTIONARIES
                USED_ARUCO_DICTIONARIES.add(dictionary_name)
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    points_3d = np.array([[width_scale - (corner[0] - center_x) * width_scale,
                                           (corner[1] - center_y) * height_scale,
                                           0.0] for corner in marker_corners[0]], dtype=np.float32)
                    if marker_id in ARUCO_PER_ID[dictionary_name]:
                        raise Exception(f"Duplicate Aruco Marker found: {dictionary_name}, {marker_id} ({self.name} -> {ARUCO_PER_ID[dictionary_name][marker_id].plane.name})")

                    print (f"found auco for {self.name}: {dictionary_name}, {marker_id}")
                    ARUCO_PER_ID[dictionary_name][marker_id] = Aruco(int(marker_id), dictionary_name, points_3d, self)


        # Keypoints
        #
        reference_img_gray = cv2.cvtColor(self.warped_reference_img, cv2.COLOR_BGR2GRAY)
        detector_name = FEATURE_DETECTOR.upper()

        if detector_name == "AKAZE":
            self.detector = cv2.AKAZE_create()
        elif detector_name == "ORB":
            self.detector = cv2.ORB_create(1500, nlevels=8)
        else:
            raise ValueError(f"Unsupported feature detector: {FEATURE_DETECTOR}")

        self.reference_keypoints, self.reference_descriptors = self.detector.detectAndCompute(reference_img_gray, None)
        if self.reference_descriptors is None or len(self.reference_keypoints) == 0:
            raise RuntimeError("Could not extract features from reference image")
        self.reference_points_3d = np.array(
            [
                [(keypoint.pt[0] - center_x) * width_scale, (keypoint.pt[1] - center_y) * height_scale, 0.0]
                for keypoint in self.reference_keypoints
            ],
            dtype=np.float32,
        )
        if BRUTEFORCE_MATCHER:
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

        # for the ones that were matched but are not matched currently, we do optical flow
        #
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
            missing_query_indices = [query_index for query_index in self.previous_tracking["points_by_query"] if query_index not in matched_query_indices]
            if not missing_query_indices:
                self.optical_flow_query_indices = set()
            else:
                previous_points = np.float32([self.previous_tracking["points_by_query"][query_index] for query_index in missing_query_indices]).reshape(-1, 1, 2)
                tracked_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.previous_tracking["frame_gray"],
                    frame_gray,
                    previous_points,
                    None,
                    winSize=FLOW_WINDOW_SIZE,
                    maxLevel=FLOW_MAX_LEVEL,
                )
                self.optical_flow_query_indices = set()
                if tracked_points is not None and status is not None:
                    frame_height, frame_width = frame_gray.shape[:2]
                    recovered_matches = list(self.good_matches)
                    recovered_keypoints = list(self.key_points_on_full_frame)
                    for query_index, point, keep, point_error in zip(missing_query_indices, tracked_points.reshape(-1, 2), status.ravel(), error.ravel()):
                        if not keep or point_error > FLOW_MAX_ERROR:
                            continue
                        x, y = point
                        if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
                            continue
                        if x < left or x >= right or y < top or y >= bottom:
                            continue
                        recovered_keypoints.append(cv2.KeyPoint(float(x), float(y), 8.0))
                        recovered_matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=len(recovered_keypoints) - 1, _distance=float(point_error)))
                        self.optical_flow_query_indices.add(query_index)
                    self.good_matches = recovered_matches
                    self.key_points_on_full_frame = recovered_keypoints

        self.previous_tracking["frame_gray"] = frame_gray.copy()
        self.previous_tracking["points_by_query"] = {
            match.queryIdx: np.array(self.key_points_on_full_frame[match.trainIdx].pt, dtype=np.float32) for match in self.good_matches
        }

        # check homography to see which matches are good
        #
        if len(self.good_matches) < MIN_MATCH_COUNT:
            self.homography, self.inlier_mask = None, []
        else:
            reference_points = np.float32([self.reference_keypoints[match.queryIdx].pt for match in self.good_matches]).reshape(-1, 1, 2)
            frame_points = np.float32([self.key_points_on_full_frame[match.trainIdx].pt for match in self.good_matches]).reshape(-1, 1, 2)
            self.homography, mask = cv2.findHomography(reference_points, frame_points, cv2.RANSAC, RANSAC_THRESHOLD)
            self.inlier_mask = [] if mask is None else mask.ravel().astype(bool).tolist()
            inlier_count = sum(self.inlier_mask)
            if inlier_count < MIN_MATCH_COUNT:
                self.homography, self.inlier_mask = None, []

        self.inlier_matches = [match for match, keep in zip(self.good_matches, self.inlier_mask) if keep] if self.inlier_mask else []

        homography_confidence = geometry_utils.rate_homography(self.homography)

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
        # if len(self.inlier_matches) < 4: # should we bring this back????
        #     return False
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
                drawing_utils.draw_pose_axes_overlay(
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
        else:
            used_index = 0
            translation_vector = np.array(translation_vectors[used_index].reshape(3), dtype=np.float64)
            rotation_matrix, _ = cv2.Rodrigues(rotation_vectors[used_index])


            if STRAIGHTEN_Z_ON_FRONT:
                z_axis = rotation_matrix[:, 2]
                z_axis = z_axis / np.linalg.norm(z_axis)
                horizontal_angle_degrees = float(np.degrees(np.arctan2(abs(z_axis[0]), abs(z_axis[2]))))
                vertical_angle_degrees = float(np.degrees(np.arctan2(abs(z_axis[1]), abs(z_axis[2]))))
                horizontal_blend = 1.0 - np.clip(
                    (horizontal_angle_degrees - STRAIGHT_ROTATION_START_ANGLE_DEGREES*2) / (STRAIGHT_ROTATION_END_ANGLE_DEGREES*2 - STRAIGHT_ROTATION_START_ANGLE_DEGREES*2),
                    0.0,
                    1.0,
                )
                vertical_blend = 1.0 - np.clip(
                    (vertical_angle_degrees - STRAIGHT_ROTATION_START_ANGLE_DEGREES) / (STRAIGHT_ROTATION_END_ANGLE_DEGREES - STRAIGHT_ROTATION_START_ANGLE_DEGREES),
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

            if True:
                translation_vector = translation_vector + rotation_matrix @ self.translation_offset
                rotation_matrix = rotation_matrix @ np.array(self.rotation_offset, dtype=np.float64)
                if frame_debug is not None:
                    debug_rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
                    debug_axis_points = np.array([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)], dtype=np.float32)
                    debug_axis_points_2d, _ = cv2.projectPoints(debug_axis_points, debug_rotation_vector, translation_vector, camera_matrix, distortion_coefficients)
                    debug_axis_points_2d = np.rint(debug_axis_points_2d.reshape(-1, 2)).astype(np.int32)
                    debug_axis_origin = tuple(debug_axis_points_2d[0])
                    cv2.line(frame_debug, debug_axis_origin, tuple(debug_axis_points_2d[1]), (0, 0, 125), 3)
                    cv2.line(frame_debug, debug_axis_origin, tuple(debug_axis_points_2d[2]), (0, 125, 0), 3)
                    cv2.line(frame_debug, debug_axis_origin, tuple(debug_axis_points_2d[3]), (125, 0, 0), 3)


            global LAST_POSES_CHECK
            if LAST_POSES_CHECK:

                is_valid = False
                for last_rotation_matrix in LAST_POSES_CHECK:
                    rotation_similarity = geometry_utils.rotation_matrix_similarity(last_rotation_matrix, rotation_matrix)
                    if rotation_similarity > 0.7:
                        is_valid = True
                if not is_valid:
                    LAST_POSES_CHECK.append(rotation_matrix)
                    return False
                else:
                    LAST_POSES_CHECK = [rotation_matrix]
            else:
                LAST_POSES_CHECK = [rotation_matrix]


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



        if DRAW_HOMOGRAPHY_OUTLINE and self.homography is not None:
            reference_height, reference_width = self.warped_reference_img.shape[:2]
            reference_corners = np.array(
                [[0.0, 0.0], [reference_width - 1.0, 0.0], [reference_width - 1.0, reference_height - 1.0], [0.0, reference_height - 1.0]],
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
        frame_with_references[reference_top:reference_top + scaled_height, :scaled_width] = stretched_reference
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

left_plane = Plane('left', './captures/left.json', rotation_offset=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                   translation_offset=(0, 0, BOX_WIDTH * 0.5), world_size=(BOX_DEPTH, BOX_HEIGHT),
                   display_color_multiplier=0.5)
right_plane = Plane('right', './captures/right.json', rotation_offset=[[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                    translation_offset=(0, 0, BOX_WIDTH * 0.5), world_size=(BOX_DEPTH, BOX_HEIGHT),
                   display_color_multiplier=0.5)
front_plane = Plane('front', './captures/front.json', rotation_offset=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    translation_offset=(0, 0, BOX_DEPTH * 0.5), world_size=(BOX_WIDTH, BOX_HEIGHT))
back_plane = Plane('back', './captures/back.json', rotation_offset=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                   translation_offset=(0, 0, BOX_DEPTH * 0.5), world_size=(BOX_WIDTH, BOX_HEIGHT))


def main():
    all_planes = [front_plane, left_plane, right_plane] #, back_plane]
    time_before_load_detection_model = time.time()
    detection_model = yolo_utils.load_detection_model(MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU, ONNX_INPUT_SIZE)
    print (f'loading yolo model took {time.time() - time_before_load_detection_model} seconds.')
    print(detection_model.describe())
    reference_column_width = max(plane.warped_reference_img.shape[1] for plane in all_planes)

    is_video_file = isinstance(CAMERA_INDEX_OR_FILE, (str, Path))

    cap = cv2.VideoCapture(CAMERA_INDEX_OR_FILE)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX_OR_FILE}.")
        sys.exit(1)

    capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {capture_width}x{capture_height}")
    if is_video_file and START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    movie_writer = None
    paused = False
    pause_after_first_frame = START_PAUSED
    debug_view = False
    recent_yolo_bounds = []
    blended_pose_result = None
    pose_kalman_filter = geometry_utils.PoseKalmanFilter(enabled=ENABLE_POSE_KALMAN)
    filtered_pose_plane_name = None

    detections = []
    combined_yolo_bounds = None
    box_overlay_opacity = 1.0

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read initial frame from webcam.")
        cap.release()
        sys.exit(1)
    frame = geometry_utils.crop_frame_to_width(frame, OUTPUT_FRAME_WIDTH)
    print(f"Working resolution: {frame.shape[1]}x{frame.shape[0]}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or not np.isfinite(fps):
        fps = DEFAULT_FPS

    if RECORD_WHEN_CAMERA and isinstance(CAMERA_INDEX_OR_FILE, (int, float)):
        frame_height, frame_width = frame.shape[:2]
        movie_writer = cv2.VideoWriter(MOVIE_FILE_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
        if not movie_writer.isOpened():
            print(f"Error: Could not open movie writer for {MOVIE_FILE_PATH}.")
            cap.release()
            sys.exit(1)
        movie_writer.write(frame)
    active_camera_matrix = create_fallback_camera_matrix(frame.shape)
    active_distortion_coefficients = np.zeros((5, 1), dtype=np.float32)
    current_frame_number = START_FRAME if is_video_file else 0
    use_current_frame = True
    step_once = False
    recent_fps_values = deque(maxlen=TIMING_AVERAGE_WINDOW)

    (fps_text_width, fps_text_height), _ = cv2.getTextSize("fps 000.0", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    while True:
        loop_start_time = time.perf_counter()

        if not paused or step_once:
            if use_current_frame:
                use_current_frame = False
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from webcam.")
                    break
                current_frame_number += 1
                frame = geometry_utils.crop_frame_to_width(frame, OUTPUT_FRAME_WIDTH)
                if movie_writer is not None:
                    movie_writer.write(frame)
            unflipped_frame = np.copy(frame)
            cv2.flip(frame, 1, frame)
            frame_preview = np.copy(frame)
            frame_height, frame_width = frame_preview.shape[:2]
            best_plane = None
            best_plane_confidence = 0.0
            aruco_found_count = 0


            if not SKIP_ALL:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                arucos_per_planes = defaultdict(list)
                for dictionary_name in USED_ARUCO_DICTIONARIES:
                    corners, marker_ids, _rejected = ARUCO_DETECTORS[dictionary_name].detectMarkers(unflipped_frame)
                    if marker_ids is not None:
                        for m, marker_id in enumerate(marker_ids.ravel()):
                            found_aruco = ARUCO_PER_ID[dictionary_name].get(marker_id, None)
                            if found_aruco:
                                aruco_found_count += 1
                                arucos_per_planes[found_aruco.plane].append(found_aruco)
                                corners[m][0][:, 0] = frame_width - corners[m][0][:, 0]
                                found_aruco.last_corners = corners[m][0]
                                if debug_view:
                                    cv2.aruco.drawDetectedMarkers(frame_preview, [corners[m]], np.array([[int(marker_id)]], dtype=np.int32))
                                edge_vectors = np.roll(found_aruco.last_corners, -1, axis=0) - found_aruco.last_corners
                                edge_lengths = np.linalg.norm(edge_vectors, axis=1)
                                normalized_edge_vectors = edge_vectors / np.maximum(edge_lengths[:, None], 1e-6)
                                angle_score = 1.0 - float(np.mean(np.abs(np.sum(normalized_edge_vectors * np.roll(normalized_edge_vectors, -1, axis=0), axis=1))))
                                side_length_score = float(np.min(edge_lengths) / max(np.max(edge_lengths), 1e-6))
                                diagonal_lengths = np.array(
                                    [
                                        np.linalg.norm(found_aruco.last_corners[2] - found_aruco.last_corners[0]),
                                        np.linalg.norm(found_aruco.last_corners[3] - found_aruco.last_corners[1]),
                                    ],
                                    dtype=np.float32,
                                )
                                diagonal_score = float(np.min(diagonal_lengths) / max(np.max(diagonal_lengths), 1e-6))
                                found_aruco.frontal_score = float(np.clip(angle_score * side_length_score * diagonal_score, 0.0, 1.0))

                if arucos_per_planes:
                    best_plane = max(arucos_per_planes, key=lambda k: max(a.frontal_score for a in arucos_per_planes[k]))
                    best_arucos = arucos_per_planes[best_plane]
                    best_plane.points_3d = np.concatenate([a.points_3d for a in best_arucos])
                    best_plane.points_2d = np.concatenate([a.last_corners for a in best_arucos])
                    best_plane_confidence = 1.0

                    success, rotation_vectors, translation_vectors, _ = cv2.solvePnPGeneric(
                        best_plane.points_3d,
                        best_plane.points_2d,
                        active_camera_matrix,
                        active_distortion_coefficients,
                        flags=cv2.SOLVEPNP_IPPE,
                    )
                    projected_bounds = None
                    if success:
                        projected_corner_points, _ = cv2.projectPoints(
                            best_plane.four_corner_points_3d,
                            rotation_vectors[0],
                            translation_vectors[0],
                            active_camera_matrix,
                            active_distortion_coefficients,
                        )
                        projected_corner_points = np.rint(projected_corner_points.reshape(-1, 2)).astype(np.int32)
                        projected_bounds = (
                            int(np.min(projected_corner_points[:, 0])),
                            int(np.min(projected_corner_points[:, 1])),
                            int(np.max(projected_corner_points[:, 0])),
                            int(np.max(projected_corner_points[:, 1])),
                        )

                    best_plane.find_matches(frame_gray, projected_bounds, concat_points=True)
                    best_plane_confidence = 1.0
                    best_plane.estimate_pose_from_matches(active_camera_matrix, active_distortion_coefficients) #, frame_debug=frame_preview)

                else: # no aruco found
                    detections = detection_model.predict(frame)
                    best_yolo_detection = yolo_utils.select_best_yolo_detection(detections)
                    if best_yolo_detection is not None:
                        recent_yolo_bounds.append(best_yolo_detection["bounds"])
                        recent_yolo_bounds = recent_yolo_bounds[-NUM_CHECK_LAST_YOLO_BOX:]
                    combined_yolo_bounds = geometry_utils.combine_detection_bounds(recent_yolo_bounds, frame.shape)

                    best_plane = None
                    for p, plane in enumerate(all_planes):
                        plane_confidence = plane.find_matches(frame_gray, combined_yolo_bounds)
                        if plane_confidence > 0.9:
                            found_pose = plane.estimate_pose_from_matches(active_camera_matrix, active_distortion_coefficients)
                            if found_pose:
                                if p != 0:
                                    all_planes.insert(0, all_planes.pop(p))
                                best_plane = plane
                                best_plane_confidence = plane_confidence
                                break

                step_once = False
                if pause_after_first_frame:
                    paused = True
                    pause_after_first_frame = False

                if best_plane is not None:
                    if filtered_pose_plane_name != best_plane.name:
                        pose_kalman_filter.reset()
                        filtered_pose_plane_name = best_plane.name
                    current_pose_blend = float(np.interp(best_plane_confidence, [0.0, 0.5, 0.65, 1.0], [0.0, 0.0, 1.0, 1.0]))
                    filtered_pose_result = pose_kalman_filter.filter_pose(best_plane.pose_result)
                    blended_pose_result = geometry_utils.blend_pose_result(blended_pose_result, filtered_pose_result, current_pose_blend)
                else:
                    filtered_pose_plane_name = None

                aruco_status_text = f"ArUco count: {aruco_found_count}"
                aruco_status_color = (0, 0, 255) if aruco_found_count == 0 else (255, 255, 255)
                cv2.putText(frame_preview, aruco_status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, aruco_status_color, 2, cv2.LINE_AA)
                cv2.putText(frame_preview, aruco_status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

        else: # paused
            pass


        # draw everything
        #
        loop_duration = max(time.perf_counter() - loop_start_time, 1e-6)
        instantaneous_fps = 1.0 / loop_duration
        recent_fps_values.append(instantaneous_fps)
        averaged_fps = float(np.mean(recent_fps_values)) if recent_fps_values else None

        if not SKIP_ALL:
            if best_plane is not None:
                drawing_utils.draw_box_overlay(frame_preview, active_camera_matrix, active_distortion_coefficients, blended_pose_result, BOX_SIZE, opacity=box_overlay_opacity)
            text_origin = (frame_preview.shape[1] - fps_text_width - 20, 20 + fps_text_height)
            cv2.putText(frame_preview, f"fps: {averaged_fps:.1f}", text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if debug_view:
                yolo_utils.draw_yolo_overlay(frame_preview, detections, combined_bounds=combined_yolo_bounds)
                if best_plane is not None:
                    best_plane.draw(frame_preview, active_camera_matrix, active_distortion_coefficients)

                frame_height, frame_width = frame_preview.shape[:2]
                total_reference_height = sum(plane.get_scaled_reference_size(reference_column_width)[1] for plane in all_planes)
                canvas_height = max(frame_height, total_reference_height)
                canvas_width = reference_column_width + frame_width
                frame_with_references = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                frame_with_references[:frame_height, reference_column_width:reference_column_width + frame_width] = frame_preview

                if best_plane is not None:
                    best_plane.draw_lines_to_reference(frame_with_references, reference_column_width, 0, best_plane_confidence)

                cv2.putText(frame_with_references, f"D: debug {'on' if debug_view else 'off'} | SPACE: pause | Q: quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, frame_with_references)
            else:
                cv2.imshow(WINDOW_NAME, frame_preview)
        else:
            cv2.imshow(WINDOW_NAME, frame_preview)

        raw_key = cv2.waitKeyEx(1)

        key = raw_key & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("d"), ord("D")):
            debug_view = not debug_view
        if key == ord(" "):
            paused = not paused

        if paused and is_video_file and raw_key in (2555904, 83):
            step_once = True

    cap.release()
    if movie_writer is not None:
        movie_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
