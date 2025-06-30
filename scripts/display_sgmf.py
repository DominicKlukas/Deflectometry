import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load both NumPy files
file1 = np.load("data/SGMF/sgmf1.npy", allow_pickle=True)
file2 = np.load("data/SGMF/sgmf3.npy", allow_pickle=True)

# Extract grayscale values from both files
image1_raw_1 = np.array([[file1[0, 0, i, j] for j in range(1080)] for i in range(1920)], dtype=np.float32)
image1_raw_2 = np.array([[file1[0, 1, i, j] for j in range(1080)] for i in range(1920)], dtype=np.float32)

image2_raw_1 = np.array([[file2[0, 0, i, j] for j in range(1080)] for i in range(1920)], dtype=np.float32)
image2_raw_2 = np.array([[file2[0, 1, i, j] for j in range(1080)] for i in range(1920)], dtype=np.float32)

# Create figure for slice visualization
fig, (ax_y, ax_x) = plt.subplots(1, 2, figsize=(12, 5))

# Initial slice positions
y_index = 960  # Middle row for Y-slice
x_index = 540  # Middle column for X-slice

# **Plot Y-slice from image1 (X = 1920 pixels, Y = grayscale values)**
line_y1, = ax_y.plot(range(1920), image1_raw_1[:, y_index], color='black', label="Camera1")
line_y2, = ax_y.plot(range(1920), image2_raw_1[:, y_index], color='blue', linestyle="dashed", label="Camera2")

ax_y.set_title(f"Y-Slice from Image 1 at row {y_index}")
ax_y.set_xlabel("X Position (0-1919)")
ax_y.set_ylabel("Grayscale Value (0-1000)")
ax_y.set_xlim(0, 1920)
ax_y.set_ylim(0, 2000)
ax_y.legend()

# **Plot X-slice from image2 (X = 1080 pixels, Y = grayscale values)**
line_x1, = ax_x.plot(range(1080), image1_raw_2[x_index, :], color='black', label="Camera1")
line_x2, = ax_x.plot(range(1080), image2_raw_2[x_index, :], color='blue', linestyle="dashed", label="Camera2")

ax_x.set_title(f"X-Slice from Image 2 at column {x_index}")
ax_x.set_xlabel("Y Position (0-1079)")
ax_x.set_ylabel("Grayscale Value (0-1000)")
ax_x.set_xlim(0, 1080)
ax_x.set_ylim(0, 2000)
ax_x.legend()

# Adjust layout for sliders
plt.subplots_adjust(bottom=0.25)

# Create sliders

position_y = (0.15, 0.1, 0.65, 0.03)
position_x = (0.15, 0.05, 0.65, 0.03)
assert isinstance(position_y, tuple) and len(position_y) == 4, "Slider position must be a tuple of length 4"
assert isinstance(position_x, tuple) and len(position_x) == 4, "Slider position must be a tuple of length 4"

ax_slider_y = plt.axes(position_y)
ax_slider_x = plt.axes(position_x)

slider_y = Slider(ax_slider_y, 'Row (Y) for Image 1', 0, 1079, valinit=y_index, valstep=1)
slider_x = Slider(ax_slider_x, 'Column (X) for Image 2', 0, 1919, valinit=x_index, valstep=1)

# Update function for sliders
def update_y(val):
    y_idx = int(slider_y.val)
    line_y1.set_ydata(image1_raw_1[:, y_idx])  # Update Y-slice (image1 from Camera1)
    line_y2.set_ydata(image2_raw_1[:, y_idx])  # Update Y-slice (image1 from Camera2)
    ax_y.set_title(f"Y-Slice from Image 1 at row {y_idx}")
    fig.canvas.draw_idle()

def update_x(val):
    x_idx = int(slider_x.val)
    line_x1.set_ydata(image2_raw_2[x_idx, :])  # Update X-slice (image2 from Camera1)
    line_x2.set_ydata(image2_raw_2[x_idx, :])  # Update X-slice (image2 from Camera2)
    ax_x.set_title(f"X-Slice from Image 2 at column {x_idx}")
    fig.canvas.draw_idle()

# Connect sliders to update functions
slider_y.on_changed(update_y)
slider_x.on_changed(update_x)

plt.show()

