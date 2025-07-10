import pyopencl as cl
import pyopencl as cl
import numpy as np
from pathlib import Path
import pytest  # If using pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from config_loader import MeasurementConfig

# Setup context and queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# Paths
KERNEL_PATH = Path(__file__).resolve().parents[1] / "kernels" / "compute_surface.cl"

# Load and build the OpenCL program
with open(KERNEL_PATH, 'r') as f:
    kernel_code = f.read()

program = cl.Program(ctx, kernel_code).build()

def test_camera_dtype():
    # Example: simple test for a debug kernel that outputs ray.L

    # Get the directory of the current Python file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the .cl kernel file
    yaml_path = os.path.join(script_dir, "../config/measurement.yaml")
    Objects = MeasurementConfig(yaml_path)
    camera_np = Objects.cameras[0]
    # Create a dummy camera (float3s and float values packed)


    camera_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=camera_np)

    # Input point
    points_np = np.array([(0.0, 0.0, 0.0)], dtype=np.float32)
    points_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=points_np)

    # Output
    output_np = np.zeros((1, 3), dtype=np.float32)
    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)

    # Run test kernel
    program.test_camera_loading(queue, (1,), None, output_buf, points_buf, camera_buf)
    cl.enqueue_copy(queue, output_np, output_buf)
    print("Ray direction (L):", output_np)

def test_compute_camera_pixel():
    # Example: Test camera pixel values for some known points
    #Points
    points = np.array([[-1.5, 0, 0, 0], [0, -0.0486997, 0, 0]], dtype=np.float32)
    pixels = np.array([[170, 518], [960, 540]], dtype=np.int32) 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "../config/measurement.yaml")
    Objects = MeasurementConfig(yaml_path)
    camera_np = Objects.cameras[0]
    
    #Input buffers
    camera_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=camera_np)
    points_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=points)

    # Output
    output_np = np.zeros((2, 2), dtype=np.int32)
    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)

    # Run test kernel
    program.test_compute_camera_pixel(queue, (2,), None, output_buf, points_buf, camera_buf)
    cl.enqueue_copy(queue, output_np, output_buf)
    np.testing.assert_array_almost_equal(output_np,pixels)

def test_screen_dtype():

    # Example: simple test for a debug kernel that outputs ray.L

    # Get the directory of the current Python file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the .cl kernel file
    yaml_path = os.path.join(script_dir, "../config/measurement.yaml")
    Objects = MeasurementConfig(yaml_path)
    screen_np = Objects.screen
    # Create a dummy camera (float3s and float values packed)
    screen_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=screen_np)


    # Run test kernel
    program.test_screen_loading(queue, (1,), None, screen_buf)


def test_SGMF_lookup():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sgmf_path = os.path.join(script_dir, "../data/SGMF/sgmf.npy")
    SGMF = np.load(sgmf_path, allow_pickle=True)
    SGMF_1 = SGMF[0]
    SGMF_2 = SGMF[1]

    pixels_1 = np.array([[1543, 1019],[375, 1019],[960, 540]], dtype=np.int32)
    pixels_2 = np.array([[445, 141],[1470, 145],[960, 540]], dtype=np.int32)


    SGMF_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SGMF_1)
    input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pixels_1)

    output_np = np.zeros((2, 3), dtype=np.int32)
    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)

    # Get the directory of the current Python file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the .cl kernel file
    yaml_path = os.path.join(script_dir, "../config/measurement.yaml")
    Objects = MeasurementConfig(yaml_path)
    screen_np = Objects.screen
    # Create a dummy camera (float3s and float values packed)
    screen_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=screen_np)

    # Run test kernel
    program.test_SGMF_mapping(queue, (3,), None, SGMF_buf, input_buf, output_buf, screen_buf)
    cl.enqueue_copy(queue, output_np, output_buf)
    print(output_np)
    """
    TODO:
        Look at the corners of the screen and detemrine the pixels for it.
        Then, ensure that they map to the corners for the SGMF.
        Also, check the middle points.
        Try to work more quickly!
    """
if __name__ == "__main__":
    test_SGMF_lookup()
