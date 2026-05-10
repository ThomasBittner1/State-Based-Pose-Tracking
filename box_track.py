from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import cv2
import numpy as np


import camera
import drawing
import geometry
import plane
import yolo


WINDOW_NAME = "Box Tracker"


@dataclass
class AppConfig:
    model_path: Path = Path("models/best.engine")
    yolo_confidence: float = 0.1
    yolo_iou: float = 0.1
    yolo_input_size: int = 640
    box_size: tuple[float, float, float] = (8.500, 13.765, 7.673)
    calibration_path: Path = Path("calibration/camera.json")
    input_source: int | str | Path = 0
    start_frame: int = 60
    start_paused: bool = False
    record_webcam: bool = False
    recording_path: Path = Path("recorded_001.mp4")
    default_fps: float = 30.0
    output_frame_width: int = 1440
    timing_average_window: int = 10
    skip_tracking: bool = False
    enable_pose_kalman: bool = True


@dataclass
class TrackerConfig:
    app: AppConfig
    plane_tracking: plane.PlaneTrackingConfig


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


def build_tracker_config():
    return TrackerConfig(
        app=AppConfig(),
        plane_tracking=plane.PlaneTrackingConfig(
            feature_detector=plane.FeatureDetector.ORB,
            bruteforce_matcher=True,
            min_match_count=8,
            ransac_threshold=4.0,
            flow_max_error=20.0,
            flow_window_size=(21, 21),
            flow_max_level=3,
            draw_homography_outline=False,
            straighten_z_on_front=True,
            straight_rotation_start_angle_degrees=5.0,
            straight_rotation_end_angle_degrees=10.0,
            yolo_bounds_history_size=4,
        ),
    )


def build_reference_planes(plane_tracking_config, box_size):
    box_width, box_height, box_depth = box_size
    aruco_registry = plane.ArucoRegistry()
    pose_history = plane.PoseHistory()

    left_plane = plane.Plane(
        'left',
        './captures/left.json',
        aruco_registry,
        pose_history,
        plane_tracking_config,
        rotation_offset=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        translation_offset=(0, 0, box_width * 0.5),
        world_size=(box_depth, box_height),
        display_color_multiplier=0.5,
    )
    right_plane = plane.Plane(
        'right',
        './captures/right.json',
        aruco_registry,
        pose_history,
        plane_tracking_config,
        rotation_offset=[[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        translation_offset=(0, 0, box_width * 0.5),
        world_size=(box_depth, box_height),
        display_color_multiplier=0.5,
    )
    front_plane = plane.Plane(
        'front',
        './captures/front.json',
        aruco_registry,
        pose_history,
        plane_tracking_config,
        rotation_offset=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        translation_offset=(0, 0, box_depth * 0.5),
        world_size=(box_width, box_height),
    )
    back_plane = plane.Plane(
        'back',
        './captures/back.json',
        aruco_registry,
        pose_history,
        plane_tracking_config,
        rotation_offset=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        translation_offset=(0, 0, box_depth * 0.5),
        world_size=(box_width, box_height),
    )

    box_width = 10.0
    box_height = box_width / front_plane.ratio
    box_depth = box_height * right_plane.ratio
    print (f'Estimated size is ({box_width:.3f}, {box_height:.3f}, {box_depth:.3f}')

    return [front_plane, left_plane, right_plane, back_plane], aruco_registry


def main():
    config = build_tracker_config()
    all_planes, aruco_registry = build_reference_planes(config.plane_tracking, config.app.box_size)
    time_before_load_detection_model = time.time()
    detection_model = yolo.load_detection_model(
        config.app.model_path,
        config.app.yolo_confidence,
        config.app.yolo_iou,
        config.app.yolo_input_size,
    )
    print (f'loading yolo model took {time.time() - time_before_load_detection_model} seconds.')
    print(detection_model.describe())
    reference_column_width = max(reference.warped_reference_img.shape[1] for reference in all_planes)

    input_source = config.app.input_source
    is_video_file = isinstance(input_source, (str, Path))
    video_capture_source = str(input_source) if isinstance(input_source, Path) else input_source

    cap = cv2.VideoCapture(video_capture_source)
    if not cap.isOpened():
        print(f"Error: Could not open input source {input_source}.")
        sys.exit(1)

    capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {capture_width}x{capture_height}")
    if is_video_file and config.app.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.app.start_frame)

    movie_writer = None
    paused = False
    pause_after_first_frame = config.app.start_paused
    debug_view = False
    recent_yolo_bounds = []
    blended_pose_result = None
    pose_kalman_filter = geometry.PoseKalmanFilter(enabled=config.app.enable_pose_kalman)
    filtered_pose_plane_name = None

    detections = []
    combined_yolo_bounds = None
    box_overlay_opacity = 1.0

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read initial frame from webcam.")
        cap.release()
        sys.exit(1)
    frame = geometry.crop_frame_to_width(frame, config.app.output_frame_width)
    print(f"Working resolution: {frame.shape[1]}x{frame.shape[0]}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or not np.isfinite(fps):
        fps = config.app.default_fps

    if config.app.record_webcam and isinstance(input_source, int):
        frame_height, frame_width = frame.shape[:2]
        movie_writer = cv2.VideoWriter(str(config.app.recording_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
        if not movie_writer.isOpened():
            print(f"Error: Could not open movie writer for {config.app.recording_path}.")
            cap.release()
            sys.exit(1)
        movie_writer.write(frame)
    calibration = camera.load_calibration(config.app.calibration_path)
    if calibration is None:
        calibration = camera.create_fallback_calibration(frame.shape)
        print(f"No camera calibration found at {config.app.calibration_path}; using fallback intrinsics.")
    else:
        print(f"Loaded camera calibration from {config.app.calibration_path}.")
    active_camera_matrix = calibration.camera_matrix
    active_distortion_coefficients = calibration.distortion_coefficients
    current_frame_number = config.app.start_frame if is_video_file else 0
    use_current_frame = True
    step_once = False
    recent_fps_values = deque(maxlen=config.app.timing_average_window)

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
                frame = geometry.crop_frame_to_width(frame, config.app.output_frame_width)
                if movie_writer is not None:
                    movie_writer.write(frame)
            unflipped_frame = np.copy(frame)
            cv2.flip(frame, 1, frame)
            frame_preview = np.copy(frame)
            frame_height, frame_width = frame_preview.shape[:2]
            best_plane = None
            best_plane_confidence = 0.0
            aruco_found_count = 0


            if not config.app.skip_tracking:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                arucos_per_planes = defaultdict(list)
                for dictionary_name in aruco_registry.used_dictionary_names:
                    corners, marker_ids, _rejected = aruco_registry.detect_markers(dictionary_name, unflipped_frame)
                    if marker_ids is not None:
                        for m, marker_id in enumerate(marker_ids.ravel()):
                            found_aruco = aruco_registry.get_marker(dictionary_name, marker_id)
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
                    best_yolo_detection = yolo.select_best_yolo_detection(detections)
                    if best_yolo_detection is not None:
                        recent_yolo_bounds.append(best_yolo_detection["bounds"])
                        recent_yolo_bounds = recent_yolo_bounds[-config.plane_tracking.yolo_bounds_history_size:]
                    combined_yolo_bounds = geometry.combine_detection_bounds(recent_yolo_bounds, frame.shape)

                    best_plane = None
                    for p, reference in enumerate(all_planes):
                        plane_confidence = reference.find_matches(frame_gray, combined_yolo_bounds)
                        if plane_confidence > 0.9:
                            found_pose = reference.estimate_pose_from_matches(active_camera_matrix, active_distortion_coefficients)
                            if found_pose:
                                if p != 0:
                                    all_planes.insert(0, all_planes.pop(p))
                                best_plane = reference
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
                    blended_pose_result = geometry.blend_pose_result(blended_pose_result, filtered_pose_result, current_pose_blend)
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

        if not config.app.skip_tracking:
            if best_plane is not None:
                drawing.draw_box_overlay(frame_preview,
                                               active_camera_matrix,
                                               active_distortion_coefficients,
                                               blended_pose_result,
                                               config.app.box_size,
                                               opacity=box_overlay_opacity)
            text_origin = (frame_preview.shape[1] - fps_text_width - 20, 20 + fps_text_height)
            cv2.putText(frame_preview, f"fps: {averaged_fps:.1f}", text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if debug_view:
                yolo.draw_yolo_overlay(frame_preview, detections, combined_bounds=combined_yolo_bounds)
                if best_plane is not None:
                    best_plane.draw(frame_preview, active_camera_matrix, active_distortion_coefficients)

                frame_height, frame_width = frame_preview.shape[:2]
                total_reference_height = sum(reference.get_scaled_reference_size(reference_column_width)[1] for reference in all_planes)
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
