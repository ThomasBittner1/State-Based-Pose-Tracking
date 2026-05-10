# State Based Pose Tracking

This project is a state-based box pose tracking system that combines multiple computer vision methods to improve 
robustness and tracking stability.

The system first scans the camera feed and checks if ArUcos are present in the image. And then it switches between
those 2 states:
- ArUco(s) visible 
- No ArUcos visible

### 1. ArUco(s) visible
It uses the ArUco markers to find the points used for solvePnP. 
Since small or distant ArUcos can introduce noticeable jitter and cause the estimated box pose to shake, 
the system additionally uses **ORB** or **Akaze** feature matching and **Optical Flow** tracking, to generate more 
feature points for the *solvePnP* pose estimation step.

### 1. No ArUcos visible

If no ArUcos are visible, for example due to occlusion or because none of the visible sides of the box has markers, 
the system switches to a fallback tracking mode using only **ORB** or **Akaze** and **Optical Flow** only. 
Also, to make the feature mapping more stable, **YOLO** is used to first isolate 
the box region from the background, since ORB/Akaze matching becomes significantly less stable in cluttered scenes.
You'll see that you are in *No-ArUco-Mode*, when it says "0 Arucos" in the top left view, and also you'll notice
that the frame rate (top right side of the screen) drops significantly.

### Post Stabalizing solvePnP results
In some situations, especially when the box is viewed almost perfectly planar to the camera, solvePnP can become unstable 
and jump between 2 rotations. To reduce this effect, the system applies an additional stabilization step that blends the 
estimated rotation toward horizontal, vertical, or both major axes when appropriate.  
Finally, a Kalman filter is applied to smooth the resulting pose estimates and reduce short-term jitter in both position and rotation.


# How to use it on any box
The mandatory part of the setup is just capturing the sides, and annotating those captures by drawing a rectangle
on the sides.  
No ArUcos are needed to be specified. The tool will automatically check in the images if there are ArUcos present and take their
measurements automatically.

### Capture the sides
First capture the sides of the box. Run the following command:

``` bash
python setup/annotate_image.py "front"
```
Hold the **front** side into the camera, as frontal as possible, and click SPACE. This creates or overwrites
*front.png* inside the captures directory. Click q to close, and then run:

``` bash
python setup/capture_image.py "front"
```
This opens the *front.jpg* image and let's you draw the rectangle by just clicking on the corners. When done, hit q to
close. This generates the *front.json* file that contains the info about the rectangle.

Repeat this process for each side by substituting **front** with the corresponding other side name 
(**left**, **right**, **back**, **bottom**, **top**). You don't need to do all of them.

Do not confuse left and right. Side names are defined in the box's local coordinate system, not relative to the screen or camera view.
(If you hold the box with the front face pointing forward, the side physically on your left corresponds to the left plane)

### Yolo
This is not mandatory, but a model that detects the box can make it more stable when no ArUco markers are visible.
Just generate a model (preferable using *Ultralytics*), that tracks the box. Class name doesn't matter in this case.
Specify that model in the app config's **model_path**, supported model types are **Ultralytics**, **ONNX** and **TensorRT**.


### Camera Calibration

Camera Calibration is not mandatory either, but it can help the results. To start, put the *setup/calibrate_pattern.png*
image on an iPad or print it on a flat, rigid surface (thick paper, cardboard, or plastic, don't use a 
paper that changes form), and run:



``` bash
python setup/calibrate_camera.py
```
Hold that pattern into the camera and hit SPACE. Repeat that 12 times by showing it from different angles.
When you hit q, he calculates the calibration from your images, and saves it into a json file. The next time you run 
*box_track.py*, it will automatically pick that calibration file.


# Challenges
Feature mapping with ORB/Akaze is extremely messy, and SolvePnP is sensitive. Alone a tiny jittering from ArUcos 
can produce an agressive jumping on the whole box.  
And adding more and more checks and stabalizing systems mades the evaluation speed suffer quickly. The biggest
bottleneck is checking/verifying homography on the ORB/Akaze features.


# Known Limitations
- If the box moves away from camera and becomes smaller, jittering becomes stronger quickly.
- Camera needs to be either high resolution, or the box needs to be close to camera for this to work stable.



