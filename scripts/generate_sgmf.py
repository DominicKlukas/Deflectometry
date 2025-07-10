import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pyopencl as cl
import time
from concurrent.futures import ThreadPoolExecutor
import argparse

class SGMFGenerator:
    """
    Generates the Simple Geometric Mapping Function (SGMF) from phase-unwrapped fringe data.

    This class encapsulates the logic and parameters used to construct the SGMF, from images
    of different wavelengths and phases, including epsilon-based filtering, wavenumber 
    indexing, and screen geometry.

    Attributes:
        data (np.ndarray): Phase-unwrapped input data, expected shape (N, H, W).
        This is computed on startup, from a given filepath.
        max_N (int): Maximum wavenumber used during fringe projection.
        eps (float): Epsilon value used for hysteresis in the floor operation when the
                    wavelengths are consolidated.
        scr_res (tuple[int, int]): Resolution of the projecting screen as (width, height).
    """

    def __init__(self, eps=0.02, screen_res=(2000, 2000), max_N = 8):
        """
        Initialize the SGMFGenerator.
        Parameters:
            max_N (int): The highest wavenumber index used in the projection.
            eps (float): Threshold used to filter out unreliable phase values.
            scr_res (tuple[int, int]): Resolution of the screen as (width, height).

        Raises:
            ValueError: If input dimensions are inconsistent or invalid.
        """
        self.eps = eps
        self.max_N = max_N
        self.scr_res = screen_res

    def save_SGMF(self, SGMF, file_path):
        np.save(file_path, SGMF)

    def gen_sgmf(self):
        """
        Generate the Simple Geometric Mapping Function (SGMF) from phase-unwrapped data.

        This method processes phase-unwrapped fringe data and computes the SGMF used to relate 
        screen coordinates to camera pixel locations using a specified threshold `epsilon`.
        This method uses OpenCL.

        Parameters:
            Uses object properties. However, requires ../kernels/compute_sgmf.cl file to be present.

        Returns:
            np.ndarray: The computed SGMF array of shape (2, 2, H, W), representing mapping in X and Y.
        """
        # Setup OpenCL
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        # Recall that OpenCL natively supports 3D arrays, but for more dimensions you have to do array flattening

        # Load the kernel code
        script_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(script_dir, "../kernels/compute_sgmf.cl")
        with open(kernel_path, "r") as f:
            kernel_code = f.read()

        program = cl.Program(ctx, kernel_code).build()

        # Get the arguments ready
        mf = cl.mem_flags
        global_size = (2*2, self.res[0], self.res[1]) # 2 cameras, 2 axes, and resolutionl
        sgmf = np.empty((2, 2, self.res[0], self.res[1]), np.int32) # camera, axis, x_res, y_res
        data = self.data.reshape(-1)
        input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, sgmf.nbytes)
        width = np.uint32(self.res[0])
        height = np.uint32(self.res[1])
        eps = np.float32(self.eps)
        max_N = np.uint32(self.max_N)
        N = np.uint32(self.N)
        scr_res_x = np.uint32(self.scr_res[0])
        scr_res_y = np.uint32(self.scr_res[1])

        # Make it happen!
        program.generate_data(queue, global_size, None, input_buf, output_buf, width, height, eps, max_N, N, scr_res_x, scr_res_y)

        # Collect the results
        cl.enqueue_copy(queue, sgmf.reshape(-1), output_buf)
        print(sgmf.shape, flush=True)
        queue.finish()

        sgmf = sgmf.reshape((2, 2, self.res[0], self.res[1]))
        print(sgmf.shape, flush=True)
        return sgmf

    def gen_sgmf_slow(self):
        """
        Generate the Simple Geometric Mapping Function (SGMF) from phase-unwrapped data.

        This method processes phase-unwrapped fringe data and computes the SGMF used to relate 
        screen coordinates to camera pixel locations using a specified threshold `epsilon`.
        This method does not use OpenCL but computes each pixel in a loop, so is significantly 
        slower than gen_sgmf.

        Parameters:
            None. Uses object parameters self.eps, self.max_N, and self.data when 
            self.apply_MPS_pixel is called.

        Returns:
            np.ndarray: The computed SGMF array of shape (2, 2, H, W), representing mapping in X and Y.
        """
        sgmf = np.zeros((2, 2, self.res[0], self.res[1]), dtype=np.float32)
        for cam in range(2):
            for ax in range(2):
                print(f"ax {ax}, cam {cam}", flush=True)
                for x in range(self.res[0]):
                    for y in range(self.res[1]):
                        sgmf[cam, ax, x, y] = self.apply_MPS_pixel(cam, ax, x, y, self.eps, max_N=self.max_N)
        return sgmf

    def apply_MPS_pixel(self, cam, ax, x, y, eps, max_N=None):
        """
        NOTE: This is used by the plot_mps() function, even if it isn't used in the final computation
        """
        if max_N is None:
            max_N = self.N
        if max_N == 0:
            max_N = 1
        k = 1 
        r = self.data[cam, ax, 0, x, y]
        for wn in range(1, max_N):
            next = self.data[cam, ax,wn,x,y]
            k = k*2
            phase_dif = np.abs(((1/k)*(next)-r)/(np.pi/2))
            n = np.floor(phase_dif+eps)
            r = next + n*np.pi
        # This will be raw phase, when the image has 2^((max_N-1) - 1) cycles. Here, max_N is strict inequality over the wavenumbers
        # that will be accepted.
        # Thus, in order to get the number of pixels, we have to convert: phase*
        r *= self.scr_res[ax]/(2*np.pi*2**(max_N - 2))
        return r

    def plot_slices(self):
        cam = 0
        """Create an interactive plot with sliders for X, Y, and Wavenumber."""
        fig, (ax_y, ax_x) = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.35)  # Adjust for 3 sliders

        # Initial slice positions
        fixed_y = self.res[1] // 2  # Middle row for Y-slice
        fixed_x = self.res[0] // 2  # Middle column for X-slice
        wavenumber = 0  # Start at first wavenumber

        # Compute initial slices
        y_slice = self.data[cam, 0, wavenumber, :, fixed_y]
        x_slice = self.data[cam, 1, wavenumber, fixed_x, :]

        print(len(y_slice), flush=True)
        print(self.res[0], flush=True)
        # Plot Y-slice
        line_y, = ax_y.plot(range(self.res[0]), y_slice, color='black', label="Y-Slice")
        ax_y.set_title(f"Y-Slice at row {fixed_y}, WN {wavenumber}")
        ax_y.set_xlabel(f"X Position (0-{self.res[0]-1})")
        ax_y.set_ylabel("Value")
        ax_y.set_xlim(0, self.res[0])
        ax_y.set_ylim(0, np.max(y_slice) * 1.2)
        ax_y.legend()

        # Plot X-slice
        line_x, = ax_x.plot(range(self.res[1]), x_slice, color='blue', label="X-Slice")
        ax_x.set_title(f"X-Slice at column {fixed_x}, WN {wavenumber}")
        ax_x.set_xlabel(f"Y Position (0-{self.res[1]-1})")
        ax_x.set_ylabel("Value")
        ax_x.set_xlim(0, self.res[1])
        ax_x.set_ylim(0, np.max(x_slice) * 1.2)
        ax_x.legend()

        # Create sliders with assertions to validate ranges
        position_y = (0.15, 0.2, 0.65, 0.03)
        position_x = (0.15, 0.15, 0.65, 0.03)
        position_wn = (0.15, 0.1, 0.65, 0.03)
        position_cam = (0.15, 0.05, 0.65, 0.03)


        assert isinstance(position_y, tuple) and len(position_y) == 4, "Slider position must be a tuple of length 4"
        assert isinstance(position_x, tuple) and len(position_x) == 4, "Slider position must be a tuple of length 4"
        assert isinstance(position_wn, tuple) and len(position_wn) == 4, "Slider position must be a tuple of length 4"
        assert isinstance(position_cam, tuple) and len(position_wn) == 4, "Slider position must be a tuple of length 4"

        ax_slider_y = plt.axes(position_y)
        ax_slider_x = plt.axes(position_x)
        ax_slider_wn = plt.axes(position_wn)
        ax_slider_cam = plt.axes(position_cam)

        slider_y = Slider(ax_slider_y, 'Row (Y)', 0, self.res[1] - 1, valinit=fixed_y, valstep=1)
        slider_x = Slider(ax_slider_x, 'Column (X)', 0, self.res[0] - 1, valinit=fixed_x, valstep=1)
        slider_wn = Slider(ax_slider_wn, 'Wavenumber', 0, self.N - 1, valinit=wavenumber, valstep=1)
        slider_cam = Slider(ax_slider_cam, 'Camera', 0, 1, valinit=cam, valstep=1)

        # Update functions
        def update(val):
            y_idx = int(slider_y.val)
            x_idx = int(slider_x.val)
            wn_idx = int(slider_wn.val)
            cam_idx = int(slider_cam.val)
            
            # Recompute slices
            y_slice = self.data[cam_idx, 0, wn_idx, :, y_idx]
            x_slice = self.data[cam_idx, 1, wn_idx, x_idx, :]
            
            # Update plots
            line_y.set_ydata(y_slice)
            ax_y.set_title(f"Y-Slice at row {y_idx}, WN {wn_idx}")
            ax_y.set_ylim(np.min(y_slice)*1.2, np.max(y_slice) * 1.2)
            
            line_x.set_ydata(x_slice)
            ax_x.set_title(f"X-Slice at column {x_idx}, WN {wn_idx}")
            ax_x.set_ylim(np.min(x_slice)*1.2, np.max(x_slice) * 1.2)
            
            fig.canvas.draw_idle()

        # Connect sliders
        slider_y.on_changed(update)
        slider_x.on_changed(update)
        slider_wn.on_changed(update)
        slider_cam.on_changed(update)

        plt.show() 

    def plot_mps(self):

        """Create an interactive plot with sliders for X, Y, epsilon, and max_N."""
        fig, (ax_y, ax_x) = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.4)  # Adjust for 4 sliders

        # Initial slice positions based on resolution
        y_index = self.res[1] // 2  # Middle row for Y-slice
        x_index = self.res[0] // 2  # Middle column for X-slice
        eps_init = 0.02  # Initial epsilon value
        max_N_init = self.N  # Default to full range
        cam = 0

        # Compute initial slices
        y_slice = np.array([self.apply_MPS_pixel(0, 0, x, y_index, eps_init, max_N_init) for x in range(self.res[0])])
        x_slice = np.array([self.apply_MPS_pixel(0, 1, x_index, y, eps_init, max_N_init) for y in range(self.res[1])])

        # Plot Y-slice
        line_y, = ax_y.plot(range(self.res[0]), y_slice, color='black', label="Y-Slice")
        ax_y.set_title(f"Y-Slice at row {y_index}")
        ax_y.set_xlabel(f"X Position (0-{self.res[0]-1})")
        ax_y.set_ylabel("Computed Value")
        ax_y.set_xlim(0, self.res[0])
        ax_y.set_ylim(0, np.max(y_slice) * 1.2)
        ax_y.legend()

        # Plot X-slice
        line_x, = ax_x.plot(range(self.res[1]), x_slice, color='blue', label="X-Slice")
        ax_x.set_title(f"X-Slice at column {x_index}")
        ax_x.set_xlabel(f"Y Position (0-{self.res[1]-1})")
        ax_x.set_ylabel("Computed Value")
        ax_x.set_xlim(0, self.res[1])
        ax_x.set_ylim(0, np.max(x_slice) * 1.2)
        ax_x.legend()

        # Create sliders with assertions
        position_y = (0.15, 0.25, 0.65, 0.03)
        position_x = (0.15, 0.2, 0.65, 0.03)
        position_eps = (0.15, 0.15, 0.65, 0.03)
        position_max_N = (0.15, 0.1, 0.65, 0.03)
        position_cam = (0.15, 0.05, 0.65, 0.03)

        assert isinstance(position_y, tuple) and len(position_y) == 4, "Slider position must be a tuple of length 4"
        assert isinstance(position_x, tuple) and len(position_x) == 4, "Slider position must be a tuple of length 4"
        assert isinstance(position_eps, tuple) and len(position_eps) == 4, "Slider position must be a tuple of length 4"
        assert isinstance(position_max_N, tuple) and len(position_max_N) == 4, "Slider position must be a tuple of length 4"

        ax_slider_y = plt.axes(position_y)
        ax_slider_x = plt.axes(position_x)
        ax_slider_eps = plt.axes(position_eps)
        ax_slider_max_N = plt.axes(position_max_N)
        ax_slider_cam = plt.axes(position_cam)

        slider_y = Slider(ax_slider_y, 'Row (Y)', 0, self.res[1] - 1, valinit=y_index, valstep=1)
        slider_x = Slider(ax_slider_x, 'Column (X)', 0, self.res[0] - 1, valinit=x_index, valstep=1)
        slider_eps = Slider(ax_slider_eps, 'Epsilon', 0.0, 0.1, valinit=eps_init, valstep=0.001)
        slider_max_N = Slider(ax_slider_max_N, 'Max N', 1, self.N, valinit=max_N_init, valstep=1)
        slider_cam = Slider(ax_slider_cam, 'Camera', 0, 1, valinit=cam, valstep=1)

        # Update functions
        def update(val):
            y_idx = int(slider_y.val)
            x_idx = int(slider_x.val)
            eps_val = slider_eps.val
            max_N_val = int(slider_max_N.val)
            cam_idx = int(slider_cam.val)

            self.eps = eps_val
            self.max_N = max_N_val
            
            # Recompute slices
            y_slice = np.array([self.apply_MPS_pixel(cam_idx, 0, x, y_idx, eps_val, max_N_val) for x in range(self.res[0])])
            x_slice = np.array([self.apply_MPS_pixel(cam_idx, 1, x_idx, y, eps_val, max_N_val) for y in range(self.res[1])])
            
            # Update plots
            line_y.set_ydata(y_slice)
            ax_y.set_title(f"Y-Slice at row {y_idx}")
            ax_y.set_ylim(np.min(y_slice)*1.2, np.max(y_slice) * 1.2)
            
            line_x.set_ydata(x_slice)
            ax_x.set_title(f"X-Slice at column {x_idx}")
            ax_x.set_ylim(np.min(x_slice)*1.2, np.max(x_slice) * 1.2)
            
            fig.canvas.draw_idle()

        # Connect sliders
        slider_y.on_changed(update)
        slider_x.on_changed(update)
        slider_eps.on_changed(update)
        slider_max_N.on_changed(update)
        slider_cam.on_changed(update)

        plt.show()


    def import_data(self, import_filepath):
        """
        Computes the phase of each pixel using the GPU (fast)
        Argument: path to the folder which contains all of your images
        Result: puts the numpy array with floats in the self.data variable
        """
        # Setup OpenCL
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        start = time.time()
        # Determine Number of Phase Shifts
        files = sorted(os.listdir(import_filepath))
        self.N = (int)(len(files)/(2*2*4)) # Number of images is 2 cameras * max_N * phase_shifts * axes = 2 * max_N * 4 * 2

        # Get the reference image to determine the resolution
        ref_img = np.asarray(cv2.imread(import_filepath + files[0], cv2.IMREAD_GRAYSCALE))
        self.res = (ref_img.shape[1], ref_img.shape[0])

        # pre-allocate the input array
#        input_images = np.empty((len(files), self.res[1], self.res[0]), dtype=np.uint8)

        full_paths = [import_filepath + f for f in files]

        def load_image(filepath):
            return cv2.imread(filepath, 0)

        with ThreadPoolExecutor() as executor:
            input_images = list(executor.map(load_image, full_paths))

#        for i, f in enumerate(files):
#            input_images[i] = cv2.imread(import_filepath + f, 0)
        input_images = np.array(input_images, dtype=np.uint8).reshape(-1)

        # pre-allocate the output array
        self.data = np.empty((2, 2, self.N, self.res[0], self.res[1]), dtype=np.float32)
        end = time.time()

        print(f"Image loading took {end - start:.2f} seconds", flush=True)

        mf = cl.mem_flags # we will need this object a lot, so this is for convenience

        start = time.time()
        # Create the buffers on the GPU
        input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_images)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, self.data.nbytes)

        # Get the directory of the current Python file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the .cl kernel file
        kernel_path = os.path.join(script_dir, "../kernels/compute_unwrap_phase.cl")

        # Read kernel source code
        with open(kernel_path, "r") as f:
            kernel_code = f.read()

        program = cl.Program(ctx, kernel_code).build()
        global_size = (2*2*self.N, self.res[0], self.res[1]) # 2 cameras, 2 axes, wavenumbers, and resolution

        program.generate_data(queue, global_size, None, input_buf, output_buf, np.uint32(self.res[0]), np.uint32(self.res[1]),  np.uint32(self.N))

        # Copy the data from the kernel buffer back to our python array
        cl.enqueue_copy(queue, self.data, output_buf)
        print(self.data.shape, flush=True)
        output_images = self.data.reshape((2, 2, self.N, self.res[0], self.res[1]))
        print(output_images.shape, flush=True)
        queue.finish()
        end = time.time()
        print(f"Image processing {end - start:.2f} seconds", flush=True)
        
if __name__ == "__main__":
    #filepath = "results/cam_1/"
    sg = SGMFGenerator()
    filepath = "output/"
    sg.import_data(filepath)
    sg.eps = 0.07
    sg.max_N = 5
    print(sg.eps, flush=True)
    print(sg.max_N, flush=True)
    sgmf = sg.gen_sgmf()
    sg.save_SGMF(sgmf, "data/SGMF/sgmf.npy")

def main():
    parser = argparse.ArgumentParser(
        description="SGMF Generator CLI — compute, interactively tune, or visualize unwrapped phase data."
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- compute subcommand ---
    interactive_parser = subparsers.add_parser('interactive', help="Compute SGMF using default parameters.")
    interactive_parser.add_argument('--input-file', required=True, help="Path to folder containing image files")
    interactive_parser.add_argument('--output-file', required=True, help="Path to save computed SGMF (.npy)")
    interactive_parser.add_argument(
    '--screen-res',
    required=True,
    type=int,
    nargs=2,
    metavar=('WIDTH', 'HEIGHT'),
    help="Screen resolution as two integers (e.g., 1920 1080)")

    # --- interactive subcommand ---
    compute_parser = subparsers.add_parser('compute', help="Interactive SGMF computation with epsilon tuning.")
    compute_parser.add_argument('--input-file', required=True, help="Path to unwrapped phase input (.npy)")
    compute_parser.add_argument('--output-file', required=True, help="Path to save computed SGMF (.npy)")
    compute_parser.add_argument('--eps', type=float, default=0.07, help="Initial epsilon value for filtering")
    compute_parser.add_argument('--max-N', type=int, default=5, help="Maximum wavenumber used in data")
    compute_parser.add_argument(
    '--screen-res',
    required=True,
    type=int,
    nargs=2,
    metavar=('WIDTH', 'HEIGHT'),
    help="Screen resolution as two integers (e.g., 1920 1080)"
    )

    # --- visualize subcommand ---
    visualize_parser = subparsers.add_parser('visualize', help="Visualize unwrapped input data.")
    visualize_parser.add_argument('--input-file', required=True, help="Path to unwrapped phase input (.npy)")

    args = parser.parse_args()

    # Handle subcommands
    if args.command == 'interactive':
        print(f"Computing SGMF from {args.input_file} to {args.output_file}. Select eps and max_N manually.")
        sg = SGMFGenerator(screen_res = args.screen_res)
        sg.import_data(args.input_file)
        sg.plot_mps()
        SGMF = sg.gen_sgmf()
        sg.save_SGMF(SGMF, args.output_file)

    elif args.command == 'compute':
        print(f"Computing SGMF from {args.input_file} to {args.output_file} with eps={args.eps}, max_N={args.max_N}.")
        eps = args.eps
        max_N = args.max_N
        sg = SGMFGenerator(max_N = max_N, eps = eps, screen_res = args.screen_res)
        sg.import_data(args.input_file)
        SGMF = sg.gen_sgmf()
        sg.save_SGMF(SGMF, args.output_file)
        

    elif args.command == 'visualize':
        print(f"Visualizing data from {args.input_file}")
        sg = SGMFGenerator()
        sg.import_data(args.input_file)
        sg.plot_slices()

#if __name__ == "__main__":
#    main()
"""
compute_sgmf.py interactive --input-file path --output-file loc
compute_sgmf.py interactive --input-file path --output-file loc --eps 0.5 --max-N 5
compute_sgmf.py visualize --input-file path
"""
