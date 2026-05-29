from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np


SUPPORTED_ARUCO_DICTIONARIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "7x7_50": cv2.aruco.DICT_7X7_50,
}


@dataclass
class Aruco:
    id: int
    dictionary_name: str
    points_3d: np.ndarray
    plane: object


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
