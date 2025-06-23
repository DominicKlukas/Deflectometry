# Deflectometry Toolkit

A modular pipeline for structured-light deflectometry, including pattern generation, calibration, phase unwrapping, geometric reconstruction, and result visualization.

---

## Features

- Generate fringe or binary patterns for screen projection
- Calibrate camera response against displayed intensity
- Perform pixel-wise phase unwrapping
- Compute screen-to-camera geometric mappings
- Visualize mappings and residual errors

---

## Project Structure

```
deflectometry-project/
├── scripts/                # Modular processing scripts
│   ├── generate_patterns.py
│   ├── calibrate_response.py
│   ├── unwrap_phase.py
│   ├── compute_geometry.py
│   └── plot_mapping.py
├── data/                   # Input/output data folders
│   ├── display_patterns/
│   ├── camera_images/
│   └── results/
├── config/                 # Camera, screen config files
│   └── camera_intrinsics.yaml
├── run_pipeline.sh         # End-to-end runner script
├── main.py                 # Optional CLI entry point
└── README.md
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourname/deflectometry-project.git
cd deflectometry-project
pip install -r requirements.txt
```

### 2. Run a Step

Example script calls

generate_pattern.py
```bash
python3 scripts/calibrate_response.py --image-dir data/display_patterns/2k_by_2k/images/image_050.png --points 0,0,1200,1200,500,500,800,800
```
calibrate_response.py

```bash
python3 scripts/calibrate_response.py --image-dir data/display_patterns/2k_by_2k/images/image_050.png --points 0,0,1200,1200,500,500,800,800
```

Or run the full pipeline:

```bash
bash run_pipeline.sh
```

---

## Configuration

Edit `config/camera_intrinsics.yaml` to define:

```yaml
camera_matrix: [...]
distortion_coeffs: [...]
screen_resolution: [1920, 1080]
```

### Image Naming

If the highest resolution images used have wavenumber N, we have 2^(N+1) images.
Images image_000 to image_4*2^N are horizontally striped, and image_(4*2^N + 1) to image_4*2^(N+1) are vertically striped.
For each wavenumber, there are four consecutive images for the four different phase shift values.

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
python scripts/unwrap_phase.py \
    --calibration data/results/calibration.json \
    --input data/camera_images/ \
    --output data/results/unwrap.npy
```

---

## Sample Output

| Step             | Output File                |
| ---------------- | -------------------------- |
| Calibration      | `results/calibration.json` |
| Phase Unwrapping | `results/unwrap.npy`       |
| Mapping          | `results/mapping.npy`      |
| Visualization    | `results/mapping_plot.png` |

---

## TODO

-

---

## License

MIT License — see LICENSE file.

---

## Questions?

Contact: `your.email@example.com`\
Or open an issue on GitHub.


