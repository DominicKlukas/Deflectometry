from PIL import Image
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pyopencl as cl
import argparse
import sys

class FringePattern:

    def __init__(self, screen_res, max_N, intensity):
        """
        Both screen_res and camera_res are tuples, (x, y), which describe the respective resolutions
        Assumes 4 is the number of images used for the phase shifting algorithm
        For each wavenumber, the image have (1/2)(2**wave_number) cycles of sinusoid in the image
        max_N: Specifies the maximum wavenumber.
        intensity: mean intensity, and intensity amplitude
        """
        self.screen_res = screen_res
        self.intensity = intensity
        self.max_N = max_N
        self.fp_imgs = None # Stands for fringe pattern images
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)


    def save_calibration_image(self, file_path):
        """
        file_path: file where the image will be stored. For example "fringe_patterns/" and then the file will be saved as 
        axis_x: True if you want the linear pattern to be along the x axis, false if you want along y
        min: The smallest intensity in the linear pattern
        max: The maximum intensity attained
        """
        max = self.intensity[0] + self.intensity[1]
        min = self.intensity[0] - self.intensity[1]
        image_x = self.generate_linear_x(max, min)
        image_y = self.generate_linear_y(max, min)
        # Image.fromarray requires that the first axis be the y axis, and the second the x axis
        result_x = Image.fromarray(image_x)
        result_y = Image.fromarray(image_y)
        result_x.save(file_path + "calibration_image_x.png")
        result_y.save(file_path + "calibration_image_y.png")


    def save_sinusoidal_images(self, file_path):
        """
        Filepath where you want to store the organized folders. Will have the following structure
        file_path/Axis/Wavenum/file
        Make sure it's in this format: file_path = "folder/", so should include the backslash
        The files will be labelled by phase index (so 0, 1, 2, 3).
        Note that this will delete the folders/files inside if they already exist.

        Side effects: Deletes all files inside the save folder
        """

        self.generate_image_arrays()

        # Ensure "results/" exists before deleting anything
        self.delete_files_inside(file_path)

        # Now create a new "results/images/" folder
        images_folder = os.path.join(file_path, "images/")
        os.makedirs(images_folder, exist_ok=True)
        print(images_folder)

        assert self.fp_imgs is not None
        for i in range(len(self.fp_imgs)):
            image_name = "image_" + str(i).zfill(3) + ".png"
            image = Image.fromarray(self.fp_imgs[i])
            image.save(images_folder + image_name)

    def generate_linear_x(self, max, min):
        """
        Populates a screen array for a linear pattern along the x axis.
        max: The maximum pixel value (intensity)
        min: The minimum pixel value (intensity)
        """
        x, y = self.screen_res
        image = np.zeros((y, x), dtype=np.uint8)
        for i in range(x):
            for j in range(y):
                image[j, i] = np.uint8((max-min)*i/x + min)
        return image

    def generate_linear_y(self, max, min):
        """
        Populates a screen array which is a linear pattern along the y axis.
        max: The maximum pixel value (intensity)
        min: The minimum pixel value (intensity)
        """
        x, y = self.screen_res
        image = np.zeros((y, x), dtype=np.uint8)
        for i in range(x):
            for j in range(y):
                image[j, i] = np.uint8((max - min)*j/y + min)
        return image

    def generate_image_arrays(self):
        """
        Populates an array of images that will be fringe patterns.
        """

        assert self.max_N is not None 
        self.fp_imgs = []
        x = 0
        for ax in range(2):
            for wn in range(self.max_N):
                print("The current wave number is " + str(wn), flush=True)
                for ph in range(4):
                    self.fp_imgs += [self.sinusoidal_image(ax, wn, self.intensity, ph)]
    
    def sinusoidal_image(self, axis, wave_number, intensity, phase):

        x, y = self.screen_res
        i_m, i_a = intensity

        kernel_code = """
        __kernel void sinusoidal_image(
            const int axis,
            const int wave_number,
            const float intensity_mean,
            const float intensity_amplitude,
            const int phase,
            const int width,
            const int height,
            __global uchar *output_image
        ) {
            int gid_x = get_global_id(0);
            int gid_y = get_global_id(1);

            if (gid_x >= width || gid_y >= height) return;

            float index = axis == 0 ? gid_x : gid_y;
            float phase_shift = phase / 4.0f;
            float freq = (float)(1 << wave_number) / (2.0f * (float)(axis == 0 ? width : height));
            float angle = 2.0f * M_PI_F * (index * freq + phase_shift);
            float val = intensity_mean + intensity_amplitude * sin(angle);

            int offset = gid_y * width + gid_x;
            output_image[offset] = convert_uchar_sat(val);
        }
        """

        # Build the kernel from string
        program = cl.Program(self.ctx, kernel_code).build()

        # Allocate output buffer
        output_np = np.empty((y, x), dtype=np.uint8)
        output_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output_np.nbytes)

        # Launch kernel
        global_size = (x, y)
        program.sinusoidal_image(
            self.queue, global_size, None,
            np.int32(axis),
            np.int32(wave_number),
            np.float32(i_m),
            np.float32(i_a),
            np.int32(phase),
            np.int32(x),
            np.int32(y),
            output_buf
        )

        # Copy result back to host
        cl.enqueue_copy(self.queue, output_np, output_buf)
        return output_np  

    def delete_files_inside(self, file_path):
        """
        Helper function to delete images inside of a file (if images are to be overwritten).

        file_path: file path to the location where images are to be stored. Should include a backslash.
        """
        if os.path.exists(file_path):
            # Loop through all items in "results/" and remove them
            for item in os.listdir(file_path):
                item_path = os.path.join(file_path, item)
                if os.path.isdir(item_path):  # Check if it's a folder
                    shutil.rmtree(item_path)  # Delete the folder and its contents
        print(f"Deleted all files inside: {file_path}") 

def main():
    parser = argparse.ArgumentParser(description="Determine Calibration Images")
    parser.add_argument('--save-dir', type=str, required=True, help='Directory where display patterns will be stored')
    parser.add_argument('--screen-resolution', type=int, nargs=2, required=True, help='Resolution of the screen on which the pattern is displayed. Give x, y.')
    parser.add_argument('--image-intensity', type=int, nargs=2, required=True, help='Image intensity, where the first argument is the mean intensity and the second is the amplitude. Must have min and max be between 0 and 255.')
    parser.add_argument('--mode', type=str, default='s', help='s for generate sinusoidal images, which will then also require the --max_N arguments, or t to generate calibration images')

    parser.add_argument('--max-N', type=int, required=False, help='Maximum wavenumber of the images will be 2^max_N')
    args = parser.parse_args()

    save_directory = args.save_dir
    intensity = args.image_intensity
    mode = args.mode
    resolution = args.screen_resolution
    max_N = args.max_N

    fp =  FringePattern(resolution, max_N, intensity)

    if mode == 't':
        fp.save_calibration_image(save_directory)
    else:
        if max_N is None:
            print("Make sure max_N argument is included.")
            sys.exit(1)
        fp.save_sinusoidal_images(save_directory)


if __name__ == "__main__":
    main()
