import pyopencl as cl
import numpy as np
from pathlib import Path
import os
from scripts.config_loader import MeasurementConfig
from scripts.generate_sgmf import SGMFGenerator
from scripts.generate_surface import SurfaceGenerator
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    parser = argparse.ArgumentParser(
        description="Use OpenCL kernel which implements a deflectometry algorithm to compute and graph surface points."
    )
    parser.add_argument('--input-image-folder', required=True, help="Path to folder containing image files")
    args = parser.parse_args()
    input_file_path = args.input_image_folder
    project_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path_name = "config/measurements.yaml"
    yaml_path = os.path.join(project_dir, yaml_path_name)
    input_file_path = os.path.join(project_dir, input_file_path)
    Objects = MeasurementConfig(yaml_path)
    screen_res = Objects.screen_resolution
    sg = SGMFGenerator(screen_res = screen_res)
    sg.import_data(input_file_path)
    sg.plot_mps()
    SGMF = sg.gen_sgmf()

    GenSurface = SurfaceGenerator()
    points = GenSurface.generate_surface(SGMF, Objects)
    GenSurface.graph_surface(points)

if __name__ == "__main__":
    main()

