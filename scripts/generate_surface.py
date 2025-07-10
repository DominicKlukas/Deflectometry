import pyopencl as cl
import pyopencl as cl
import numpy as np
from pathlib import Path
import os
from config_loader import MeasurementConfig
from generate_sgmf import SGMFGenerator
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_surface(SGMF, Objects):
    # Setup context and queue (for OpenCL)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    # Load the OpenCL kernel code
    KERNEL_PATH = Path(__file__).resolve().parents[1] / "kernels" / "compute_surface.cl"
    with open(KERNEL_PATH, 'r') as f:
        kernel_code = f.read()
    program = cl.Program(ctx, kernel_code).build()

    # Load the Camera, Screen, and Search Param objects, and create buffers for them
    camera_1_np = Objects.cameras[0]
    camera_2_np = Objects.cameras[1]
    screen_np = Objects.screen
    search_params = Objects.search_params
    search_origins = Objects.search_origins
    camera_1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=camera_1_np)
    camera_2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=camera_2_np)
    screen_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=screen_np)
    search_params_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=search_params)
    search_origins_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=search_origins)
    
    SGMF_1_np = SGMF[0]
    SGMF_2_np = SGMF[1]
    SGMF_1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SGMF_1_np)
    SGMF_2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SGMF_2_np)

    output_np = np.empty_like(search_origins)
    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)

    n = len(output_np)

    # Run Kernel
    program.compute_surface(queue, (n,), None, search_origins_buf, SGMF_1_buf, SGMF_2_buf, output_buf, camera_1_buf, camera_2_buf, screen_buf, search_params_buf)
    # Copy Results from GPU buffer
    cl.enqueue_copy(queue, output_np, output_buf)

    # Process data
    output_np = np.delete(output_np, 3, axis=1)
    output_np = output_np[output_np[:, 2] != search_params["error_value"]]
    return output_np

def graph_surface(points):
    """
    Plot the data
    """
    # Example: list of (x, y, z) points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', s=40)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim([-1, 1])
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Use OpenCL kernel which implements a deflectometry algorithm to compute and graph surface points."
    )
    parser.add_argument('--input-file', required=True, help="Path to folder containing image files")
    args = parser.parse_args()
    input_file_path = args.input_file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "../config/measurement.yaml")
    parent_dir = os.path.join(script_dir, "..")
    input_file_path = os.path.join(parent_dir, input_file_path)
    Objects = MeasurementConfig(yaml_path)
    screen_res = Objects.screen_resolution
    sg = SGMFGenerator(screen_res = screen_res)
    sg.import_data(input_file_path)
    sg.plot_mps()
    SGMF = sg.gen_sgmf()
    points = generate_surface(SGMF, Objects)
    graph_surface(points)

if __name__ == "__main__":
    main()
