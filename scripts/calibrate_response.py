from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
from matplotlib.widgets import Slider

def plot_calibration_intensities(x_arrays, y_arrays, colors):
    """
    x_arrays: contains the data to be graphed on the x axis
    y_arrays: data to be graphed on the y axis
    colors: colors for the different lines
    """

    x_min = min([x.min() for x in x_arrays])
    x_max = max([x.max() for x in x_arrays])
    y_min = min([y.min() for y in y_arrays])
    y_max = max([y.max() for y in y_arrays])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)  # Leave space for sliders

    # Plot all datasets with different colors
    for i, (x, y, color) in enumerate(zip(x_arrays, y_arrays, colors)):
        ax.plot(x, y, color=color, label=f"Dataset {i + 1}")

    
    # Initial slope and intercept
    m_init, b_init = (y_max - y_min)/(x_max - x_min), 0

    # Create initial line controlled by sliders
    x_line = np.linspace(x_min, x_max, 100)
    y_line = m_init * x_line + b_init
    (line_plot,) = ax.plot(x_line, y_line, "r--", label="Adjustable Line")

    # Labels and legend
    ax.set_xlabel("Screen Pixel Intensity")
    ax.set_ylabel("Camera Pixel Intensity")
    ax.set_title("Image Intensities")
    ticks = np.linspace(x_min, x_max, 7)
    # compute labels in the 0–255 range
    labels = (ticks / (x_max - x_min) * 255).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend()
    position_m = (0.2, 0.1, 0.65, 0.03)
    position_b = (0.2, 0.05, 0.65, 0.03)
    assert isinstance(position_m, tuple) and len(position_m) == 4, "Slider position must be a tuple of length 4"
    assert isinstance(position_b, tuple) and len(position_b) == 4, "Slider position must be a tuple of length 4"
    # Add sliders
    ax_m = plt.axes(position_m)
    ax_b = plt.axes(position_b)
    slider_m = Slider(ax_m, "Slope (m)", -4*m_init, 4*m_init, valinit=m_init)
    slider_b = Slider(ax_b, "Intercept (b)", -175, 350, valinit=b_init)
    # Display equation dynamically
    text_box = ax.text(0.05, 0.9, f"y = {m_init:.2f}x + {b_init:.2f}", transform=ax.transAxes, fontsize=12)
    # Update function for sliders
    def update(val):
        m = slider_m.val
        b = slider_b.val
        line_plot.set_ydata(m * x_line + b)  # Update line data
        text_box.set_text(f"y = {m:.2f}x + {b:.2f}")  # Update equation
        fig.canvas.draw_idle()
    slider_m.on_changed(update)
    slider_b.on_changed(update)
    # Show plot
    plt.show()

def analyze_calibration_image(file_path, pixel_arrays, colors):
    """
    pixel_arrays: a list of lists. Each list is 4 long.
    It should have 4 pixels: min_x, min_y, max_x, max_y, where the min are co-ordinates of a first point,
    and the max are the co-ordinates of a second point, that form a line along which we have the desired
    intensity values, in the image.
    colors: Should be a list of color strings that can be used by matplotlib to display the colors of each of them
    file_path: location of the image.
    """
    # Later, perhaps implement so that you can have other axis than x as horizontal axis in the plot
    # 1. Open the image using Pillow
    image = Image.open(file_path)

    x_arrays = []
    y_arrays = []
    for p_arr in pixel_arrays:
        # 2. Convert the Pillow Image to a NumPy array
        image_array = np.array(image)
        min_x, min_y, max_x, max_y = p_arr
        x_values = np.array(range(abs(max_x - min_x)))
        y_values = []
        for i in range(len(x_values)):
            x = min_x + np.sign(max_x - min_x)*i
            y = (int)((x-min_x)*(max_y - min_y)/(max_x - min_x)) + min_y
            if len(image_array.shape) > 2:
                y_values += [image_array[y][x][0]]
            else:
                y_values += [image_array[y][x]]
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        x_arrays += [x_values]
        y_arrays += [y_values]
    plot_calibration_intensities(x_arrays, y_arrays, colors)

def main():
    parser = argparse.ArgumentParser(description="Determine Calibration Images")
    parser.add_argument('--image-dir', type=str, required=True, help='Directory with captured images')
    parser.add_argument('--points', type=str, required=True, help='Points, in the form xmin,ymin,xmax,ymax, where the min point is a black pixel, the max point is a white pixel, and the line between the two increases linearly between them in the calibration image, in groups of 4')
    args = parser.parse_args()

    points = [int(x) for x in args.points.split(',')]
    points = [points[i:i+4] for i in range(0, len(points),4)]

    stock = ['b', 'g', 'r', 'c', 'k', 'm']
    colors = stock[:len(points)]

    analyze_calibration_image(args.image_dir, points, colors)

if __name__ == "__main__":
    main()
