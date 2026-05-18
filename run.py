from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import cv2
import numpy as np


import aruco
import camera
import geometry
import plane
import yolo


WINDOW_NAME = "Box Tracker"


@dataclass
class AppConfig:
    input_source: int | str | Path = 0 # this is the camera index or video path
    camera_calibration_path: Path = Path("calibration/camera.json")
    model_path: Path = Path(r"C:\ComputerVision\_Datasets_\tb_dataManager\box_tracker_tracker\train\weights\best.engine")
    default_fps: float = 30.0

    # feature matching / optical flow:
    feature_detector: plane.FeatureDetectorName = "ORB"
    bruteforce_matcher: bool = True
    min_match_count: int = 8
    ransac_threshold: float = 4.0
    flow_max_error: float = 20.0
    flow_window_size: tuple[int, int] = (21, 21)
    flow_max_level: int = 3

    # post stabalizing:
    enable_pose_kalman: bool = True
    enable_pose_outlier_detector: bool = True
    straighten_z_on_front: bool = False
    straight_rotation_start_angle_degrees: float = 5.0
    straight_rotation_end_angle_degrees: float = 10.0

    # debug:
    fps_display_average_window: int = 10
    debug_timing_log_interval: int = 60
    debug_recording_path: Path | None = None # Path("recorded_kalman2.mp4")
    debug_recording_include_overlays: bool = True
    draw_non_kalman_results: bool = True
    debug_print_timing: bool = False
    video_start_frame: int = 0

    # yolo:
    yolo_bounds_history_size: int = 4
    yolo_confidence: float = 0.1
    yolo_iou: float = 0.5
    yolo_input_size: int = 640


class PoseOutlierDetector:
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


def add_timing(frame_timings, name, start_time):
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    frame_timings[name] = frame_timings.get(name, 0.0) + duration_ms


def format_timing_summary(timing_history):
    parts = []
    for name in ("yolo", "match", "pose", "draw_box", "frame"):
        samples = timing_history.get(name)
        if samples:
            parts.append(f"{name}: {np.mean(samples):.1f} ms")
    return " | ".join(parts)


def draw_planes_overlay(
    frame,
    all_planes,
    camera_matrix,
    distortion_coefficients,
    pose_result,
    box_color=(255, 255, 255),
    box_thickness=3,
    draw_label=True,
):
    for plane in all_planes:
        plane.draw(
            frame,
            camera_matrix,
            distortion_coefficients,
            pose_result=pose_result,
            skip_if_not_visible=True,
            box_color=box_color,
            box_thickness=box_thickness,
            draw_label=draw_label,
        )


def build_reference_planes(config):
    aruco_registry = aruco.ArucoRegistry()
    pose_outlier_detector = PoseOutlierDetector()

    left_plane = plane.Plane('left', './captures/left.json', aruco_registry, pose_outlier_detector, config)
    right_plane = plane.Plane('right', './captures/right.json', aruco_registry, pose_outlier_detector, config)
    front_plane = plane.Plane('front', './captures/front.json', aruco_registry, pose_outlier_detector, config)
    back_plane = plane.Plane('back', './captures/back.json', aruco_registry, pose_outlier_detector, config)

    box_width = 10.0
    box_height = box_width / front_plane.ratio
    box_depth = box_height * right_plane.ratio
    box_size = (box_width, box_height, box_depth)
    print (f'Estimated size is ({box_size[0]:.3f}, {box_size[1]:.3f}, {box_size[2]:.3f}')

    front_plane.compute_feature_correspondences((box_width, box_height),
                                                rotation_offset=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation_offset=(0, 0, box_depth * 0.5))
    back_plane.compute_feature_correspondences((box_width, box_height),
                                               rotation_offset=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation_offset=(0, 0, box_depth * 0.5))
    left_plane.compute_feature_correspondences((box_depth, box_height),
                                                rotation_offset=[[0, 0, -1], [0, 1, 0], [1, 0, 0]], translation_offset=(0, 0, box_width * 0.5))
    right_plane.compute_feature_correspondences((box_depth, box_height),
                                                rotation_offset=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]], translation_offset=(0, 0, box_width * 0.5))

    return [front_plane, left_plane, right_plane, back_plane, back_plane], aruco_registry


def main():
    config = AppConfig(enable_pose_kalman=True, straighten_z_on_front=False)
    all_planes, aruco_registry = build_reference_planes(config)
    time_before_load_detection_model = time.time()
    detection_model = yolo.load_detection_model(
        config.model_path,
        config.yolo_confidence,
        config.yolo_iou,
        config.yolo_input_size,
    )
    print (f'loading yolo model took {time.time() - time_before_load_detection_model} seconds.')
    print(detection_model.describe())
    reference_column_width = max(plane.warped_reference_img.shape[1] for plane in all_planes)

    input_source = config.input_source
    is_video_file = isinstance(input_source, (str, Path))
    video_capture_source = str(input_source) if isinstance(input_source, Path) else input_source

    cap = cv2.VideoCapture(video_capture_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    if not cap.isOpened():
        print(f"Error: Could not open input source {input_source}.")
        sys.exit(1)

    capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {capture_width}x{capture_height}")
    if is_video_file and config.video_start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.video_start_frame)

    movie_writer = None
    paused = False
    pause_after_first_frame = False
    debug_view = False
    recent_yolo_bounds = []
    blended_pose_result = None
    pose_kalman_filter = geometry.PoseKalmanFilter(enabled=config.enable_pose_kalman)
    filtered_pose_plane_name = None

    detections = []
    combined_yolo_bounds = None

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read initial frame from webcam.")
        cap.release()
        sys.exit(1)
    print(f"Working resolution: {frame.shape[1]}x{frame.shape[0]}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or not np.isfinite(fps):
        fps = config.default_fps

    if config.debug_recording_path is not None and isinstance(input_source, int):
        debug_recording_path = Path(config.debug_recording_path)
        frame_height, frame_width = frame.shape[:2]
        movie_writer = cv2.VideoWriter(str(debug_recording_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
        if not movie_writer.isOpened():
            print(f"Error: Could not open movie writer for {debug_recording_path}.")
            cap.release()
            sys.exit(1)
        if not config.debug_recording_include_overlays:
            movie_writer.write(frame)
    calibration = camera.load_calibration(config.camera_calibration_path, (capture_width, capture_height))
    if calibration is None:
        calibration = camera.create_fallback_calibration(frame.shape)
        print(f"No matching camera calibration found at {config.camera_calibration_path}; using fallback intrinsics.")
    else:
        print(f"Loaded camera calibration from {config.camera_calibration_path}.")
    active_camera_matrix = calibration.camera_matrix
    active_distortion_coefficients = calibration.distortion_coefficients
    current_frame_number = config.video_start_frame if is_video_file else 0
    use_current_frame = True
    step_once = False
    recent_fps_values = deque(maxlen=config.fps_display_average_window)
    timing_history = defaultdict(lambda: deque(maxlen=config.fps_display_average_window))
    frame_count = 0

    (fps_text_width, fps_text_height), _ = cv2.getTextSize("fps 000.0", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    while True:
        loop_start_time = time.perf_counter()
        frame_timings = {}
        should_record_current_frame = False

        if not paused or step_once:
            should_record_current_frame = True
            if use_current_frame:
                use_current_frame = False
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from webcam.")
                    break
                current_frame_number += 1
                if movie_writer is not None and not config.debug_recording_include_overlays:
                    movie_writer.write(frame)
            unflipped_frame = np.copy(frame)
            cv2.flip(frame, 1, frame)
            frame_preview = np.copy(frame)
            frame_height, frame_width = frame_preview.shape[:2]
            best_plane = None
            best_plane_confidence = 0.0
            aruco_found_count = 0


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

                stage_start = time.perf_counter()
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
                add_timing(frame_timings, "pose", stage_start)

                stage_start = time.perf_counter()
                best_plane.find_matches(frame_gray, projected_bounds, concat_points=True)
                add_timing(frame_timings, "match", stage_start)
                best_plane_confidence = 1.0
                stage_start = time.perf_counter()
                best_plane.estimate_pose_from_matches(active_camera_matrix, active_distortion_coefficients) #, frame_debug=frame_preview)
                add_timing(frame_timings, "pose", stage_start)

            else: # no aruco found
                stage_start = time.perf_counter()
                detections = detection_model.predict(frame)
                best_yolo_detection = yolo.select_best_yolo_detection(detections)
                add_timing(frame_timings, "yolo", stage_start)
                if best_yolo_detection is not None:
                    recent_yolo_bounds.append(best_yolo_detection["bounds"])
                    recent_yolo_bounds = recent_yolo_bounds[-config.yolo_bounds_history_size:]
                combined_yolo_bounds = geometry.combine_detection_bounds(recent_yolo_bounds, frame.shape)

                best_plane = None
                stage_start = time.perf_counter()
                for p, plane in enumerate(all_planes):
                    plane_confidence = plane.find_matches(frame_gray, combined_yolo_bounds)
                    if plane_confidence > 0.9:
                        add_timing(frame_timings, "match", stage_start)
                        stage_start = time.perf_counter()
                        found_pose = plane.estimate_pose_from_matches(active_camera_matrix, active_distortion_coefficients)
                        add_timing(frame_timings, "pose", stage_start)
                        if found_pose:
                            if p != 0:
                                all_planes.insert(0, all_planes.pop(p))
                            best_plane = plane
                            best_plane_confidence = plane_confidence
                            break
                        stage_start = time.perf_counter()
                else:
                    add_timing(frame_timings, "match", stage_start)

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

            if not config.draw_non_kalman_results:
                aruco_status_text = f"ArUco count: {aruco_found_count}"
                aruco_status_color = (0, 0, 255) if aruco_found_count == 0 else (255, 255, 255)
                cv2.putText(frame_preview, aruco_status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, aruco_status_color, 2, cv2.LINE_AA)
                cv2.putText(frame_preview, aruco_status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

        else: # paused
            pass


        # draw everything
        #
        averaged_fps = float(np.mean(recent_fps_values)) if recent_fps_values else 0.0

        if best_plane is not None:
            stage_start = time.perf_counter()
            if config.draw_non_kalman_results:
                draw_planes_overlay(
                    frame_preview,
                    all_planes,
                    active_camera_matrix,
                    active_distortion_coefficients,
                    best_plane.pose_result,
                    box_color=(0, 255, 255),
                    box_thickness=2,
                    draw_label=False,
                )
            draw_planes_overlay(
                frame_preview,
                all_planes,
                active_camera_matrix,
                active_distortion_coefficients,
                blended_pose_result,
                draw_label=not config.draw_non_kalman_results,
            )
            add_timing(frame_timings, "draw_box", stage_start)
        if not config.draw_non_kalman_results:
            text_origin = (frame_preview.shape[1] - fps_text_width - 20, 20 + fps_text_height)
            cv2.putText(frame_preview, f"fps: {averaged_fps:.1f}", text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if debug_view:
            yolo.draw_yolo_overlay(frame_preview, detections, combined_bounds=combined_yolo_bounds)
            if best_plane is not None:
                if config.draw_non_kalman_results:
                    best_plane.draw(frame_preview, active_camera_matrix, active_distortion_coefficients, pose_result=best_plane.pose_result,
                        draw_label=False, draw_axes=True, box_color=(0, 255, 255), box_thickness=2)
                best_plane.draw(frame_preview, active_camera_matrix, active_distortion_coefficients, draw_axes=True)

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

        if (
            movie_writer is not None
            and config.debug_recording_include_overlays
            and should_record_current_frame
        ):
            movie_writer.write(frame_preview)

        add_timing(frame_timings, "frame", loop_start_time)
        frame_duration_ms = frame_timings["frame"]
        if frame_duration_ms > 0.0:
            recent_fps_values.append(1000.0 / frame_duration_ms)
        for timing_name, duration_ms in frame_timings.items():
            timing_history[timing_name].append(duration_ms)
        frame_count += 1
        if config.debug_print_timing and frame_count % config.debug_timing_log_interval == 0:
            timing_summary = format_timing_summary(timing_history)
            if timing_summary:
                print(timing_summary)

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
        print(f"Recorded webcam video to {Path(config.debug_recording_path).resolve()}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
