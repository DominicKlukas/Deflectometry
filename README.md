# Deflectometry Toolkit

A series of python scripts for structured-light deflectometry in blender, including pattern generation for screens, calibrating screen intensity, a phase unwrapping algorithm, a geometric reconstruction algorithm, and result visualization.

---

## Features

- Generate the patterns for screen projection required by the algorithm
- A tool to enable you to calibrate the projection screen brightness for optimal camera sensor response
- Intermediate step allows you to calibrate noise removal and maximum wavelength used in phase unwrapping algorithm which computes a mapping between camera sensor pixels and screen image pixels from which they originate. In a UI, the phase unwrapping is graphed with respect to user selected noise parameter and maximum wavelength settings. The stored phase unwrapped-data then proceeds with the selected settings.
- Outputs a 3D surface map.
- Computed with OpenCL GPU parallel processing, massively reducing compute time

---

## Project Structure

```
deflectometry-project/
├── scripts/                # Processing scripts
│   ├── calibrate_response.py
│   ├── config_loader.py
│   ├── display_sgmf.py
│   ├── generate_pattern.py
│   ├── generate_sgmf.py
│   └── generate_surface.py
├── data/                   
│   ├── display_patterns    # Folder to store display patterns (not used by script, but by deflectometry setup)
├── images/                 # Good place to store captured camera images
├── config/
│   └── measurements.yaml   # Camera/Screen measurements used by computations
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
(Show an image of the pixels, and give them names)
3. Run the following script.
```bash
python scripts/calibrate_response.py --image-dir data/display_patterns/calibration_images/captured_calibration_image_x.png --points 375,1010,1542,1010
```
4. Determine a region where the intensity response is relatively linear. Then, determine the average intensity and amplitude which sets the bounds within this linear region.
(Show an image of the calibration)

#### Screen Pattern Generation

Given the mean intensity I and the amplitude A, run the following command:
```bash
python scripts/generate_pattern.py --save-dir data/display_patterns/blender_screen/ --screen-resolution 2000 2000 --image-intensity I=50 A=25 --mode s --max-N 8
```

#### Taking Camera Measurements

Edit `config/measurements.yaml` with the following parameters:
(Show a well documented image, with images of screen, camera, and rectangle measurements).

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

## Example Usage

```bash
python scripts/generate_surface.py --input-image-folder images/sphere/
```

---

## Other Useful Standalone Script Functions

Example script calls

### generate_pattern.py

This generates image patterns to be displayed on the screen.

TODO

### calibrate_response.py

The mappings are fundamentally computed using screen pixel intensity values on reflected off of a surface captured by a camera. Consequentially, the mapping between intensity values measured by the camera for a corresponding intensity value on the screen is very important. In particular, the sinusoidal images should have a maximum and minimum intensity in a range where the measured camera intesnity responds linearly to changes in screen pixel intensity. A good way to ensure this is to take an image of the screen as it displays linearly increasing pixel intensity.

The following script analyzes such an image by requiring a list of lines ((x1, y1, x2, y2) are the end points) whose intensity is then displayed in a graph which can then be viewed, to determine a region where the intensity increase is linear.

```bash
python3 scripts/calibrate_response.py --image-dir data/blender/Calibration_X_result.png --points 424,0,1491,0,375,1015,1544,1015
```

### generate_sgmf.py

You may want to generate the Simple Geometric Mapping Function on its own and store it as a .npy file, since it is smaller than the underlying image files.

You can use the "interactive" (which lets you visually see the effects on SGMF noise when you adjust algorithm parameters, "espilon" and "N").
```bash
python3 scripts/generate_sgmf.py interactive --screen-res 2000 2000 --input-file output/ --output-file data/SGMF/sgmf.npy
```
Alternatively, if you have pre-determined SGMF parameters in mind, you can compute it right away.
```bash
python3 scripts/generate_sgmf.py compute --screen-res 2000 2000 --input-file output/ --output-file data/SGMF/sgmf.npy --eps 0.07 --max-N 5
```
The visualize function lets you take a look at your raw data, after it has been phase-unwrapped.
```bash
python3 scripts/generate_sgmf.py visualize --input-file output/ 
```
---
