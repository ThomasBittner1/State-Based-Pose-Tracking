This project is a state-based box pose tracking system that combines multiple computer vision methods to improve robustness and tracking stability.

The pipeline first checks for visible ArUco markers. If markers are detected, they are used for pose estimation and tracking. However, small or distant ArUcos can introduce noticeable jitter, causing the estimated box pose to shake. To stabilize the result, the system additionally uses ORB feature matching and optical flow tracking to generate more feature points for the solvePnP pose estimation step.

If no ArUcos are visible, for example due to occlusion or because the visible side of the box has no markers, the system switches to a fallback tracking mode using ORB and optical flow only. In this case, YOLO is used to first isolate the box region from the background, since ORB matching becomes significantly less stable in cluttered scenes.

In some situations, especially when the box is viewed almost perfectly planar to the camera, solvePnP can become unstable and jump between 2 rotations. To reduce this effect, the system applies an additional stabilization step that blends the estimated rotation toward horizontal, vertical, or both major axes when appropriate.

Finally, a Kalman filter is applied to smooth the resulting pose estimates and reduce short-term jitter in both position and rotation.



