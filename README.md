# Deflectometry Toolkit

A python script for structured-light deflectometry in blender, including pattern generation for screens, calibrating screen intensity, a phase unwrapping algorithm, a geometric reconstruction algorithm, and result visualization.

(Image showing how it works)

---

## Features

- Generate the patterns for screen projection required by the algorithm
- A tool to enable you to calibrate the projection screen brightness for optimal camera sensor response
- Intermediate step with a GUI allows you to calibrate noise removal and maximum wavelength used in the phase unwrapping algorithm. (The phase unwrapping algorithm computes a map between pixels in the captured images and original screen, essentially tracking rays of light from the screen to the camera).
- Outputs a graph of the 3D surface map.
- Each step uses OpenCL GPU parallel processing, massively reducing compute time.

---

## Project Structure

```
deflectometry-project/
├── scripts/                # Processing scripts
│   ├── calibrate_response.py   # Tool to help calibrate screen brightness
│   ├── config_loader.py    # Loads and converts measurements to datatypes used by algorithm
│   ├── display_sgmf.py     # Special script to visualize a Simple Geometric Mapping Function
│   ├── generate_pattern.py # Generates images that are projected by screen in deflectometry setup
│   ├── generate_sgmf.py    # Runs phase unwrapping algorithm to create camera-screen "sgmf" map
│   └── generate_surface.py # Computes the surface from SGMF and measurement data
├── data/                   
│   └── display_patterns    # Folder to store display patterns (for deflectometry setup, not scripts)
├── images/                 # Good place to store captured camera images from deflectometry setup
├── config/
│   └── measurements.yaml   # Camera/Screen measurements used by computations
├── kernels/                # Stores OpenCL kernel code
│   ├── compute_sgmf.cl     # Computes SGMF
│   ├── compute_surface.cl  # Computes surface 
│   └── compute_unwrap_phase.cl   # Unwraps phase
├── main.py                 # CLI entry point
└── README.md
```

---

## Installation

### 1. Clone and Install

```bash
git clone https://github.com/DominicKlukas/deflectometry-project.git
cd deflectometry-project
pip install -r requirements.txt
```
---

## Usage

### Generate Patterns to be Displayed on the Screen


#### Screen Intensity Calibration
Run the following command.

```bash
python scripts/generate_pattern.py --save-dir data/display_patterns/calibration_images/ --screen-resolution 2000 2000 --image-intensity
 127 127 --mode t
```
In particular, enter the screen resolution and the location to store the calibration image.

1. Display your image on your screen, and capture an image with your camera as the screen reflects off of a flat mirror.
2. In the capture image, determine the pixels of the endpoints of lines going from 0 intensity to full intensity.
3. Run the following script.
```bash
python scripts/calibrate_response.py --image-dir data/display_patterns/calibration_images/captured_calibration_image_x.png --points 375,1010,1542,1010
```
4. Determine a region where the intensity response is relatively linear. Then, determine the average intensity and amplitude which sets the bounds within this linear region.



#### Screen Pattern Generation

Given the mean intensity I and the amplitude A, run the following command:
```bash
python scripts/generate_pattern.py --save-dir data/display_patterns/blender_screen/ --screen-resolution 2000 2000 --image-intensity I=50 A=25 --mode s --max-N 8
```

#### Taking Camera Measurements

Edit `config/measurements.yaml`, entering the screen and camera properties directly from blender (rotation, translation, focal length, aspect ratio (must be calculated), sensor width. Also, specify the search parameters: the bottom back left, and top front right corners of the rectangular prism that contains the surace, the resolution of the points within the rectangle you would like to search, and an error parameter which determines what error between the two camera measurements is acceptable for a measurement to be considered valid. 

#### Rendering the images
Ensure that the correct images are being displayed, as an animation, in the screen.
Ensure that the folder outputs the images to the desired directory.
Render the animation. Two sets of images will be captured: one for each camera.

#### Calibrating the phase unwrapping and generating the surface
Run the following command.
```bash
python main.py --input-image-folder images/sinusoidal/
```

1. When the phase unwrapping calibraiton screen comes up, adjust in the following way.
2. First, increase epsilon until the following lines disappear.
3. Next, decrease N until the lines at the top disapper.
4. Keep N as high as possible to decrease noise.
5. Check both cameras, and other x/y values to ensure the calibration works well for all x and y. Then, close the window. The surface generation algorithm will proceed with the last selected values.

Enjoy the results! You may need to adjust the search area/camera positions if no matching points are found.

---

## Requirements

- Python 3.9+
- NumPy, OpenCV, Matplotlib
- (Optional) PyTorch or SciPy if needed

Install with:

```bash
pip install -r requirements.txt
```

---

