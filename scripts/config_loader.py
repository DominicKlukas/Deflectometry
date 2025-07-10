# config_loader.py

from numpy._core.numeric import dtype
import yaml
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# Create a dummy camera (float3s and float values packed)
camera_dtype = np.dtype([
    ("focal_point", np.float32, (4,)),
    ("origin", np.float32, (4,)),
    ("u", np.float32, (4,)),
    ("v", np.float32, (4,)),
    ("N", np.float32, (4,)), ("d", np.float32),
    ("__pad", np.float32),
    ("res", np.uint32, (2,))
])

screen_dtype = np.dtype([
    ("u", np.float32, (4,)),
    ("v", np.float32, (4,)),
    ("origin", np.float32, (4,))
])

search_params_dtype = np.dtype([
    ('d', np.float32, (4,)), 
    ('n', np.int32), 
    ('eps', np.float32),
    ('error_value', np.float32)
])

def pad_vector(vector):
    return np.pad(vector, (0, 1), mode='constant', constant_values=0)

def rotation_matrix_xyz(rotation_vector):
    rx, ry, rz = rotation_vector
    cx, cy, cz = np.cos(np.radians([rx, ry, rz]))
    sx, sy, sz = np.sin(np.radians([rx, ry, rz]))
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])
    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]])
    Rotation_Matrix = Rz@Ry@Rx 
    return Rotation_Matrix  # XYZ order

if __name__ == "__main__":
    vector = np.array([0, 0, -1], dtype=np.float32)
    rot = np.array([-29, 0, 0], dtype=np.float32)
    rot_mat = rotation_matrix_xyz(rot)
    result = rot_mat@vector
    print(np.cos(29*np.pi/180))
    print(result)

class MeasurementConfig:
    def __init__(self, path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cameras = []
        for cam in cfg["cameras"]:
            self.cameras += [self.load_camera(cam)]

        self.screen = self.load_screen(cfg["screen"])
        
        self.search_origins, self.search_params = self.load_search(cfg["search_region"])
        self.screen_resolution = cfg["screen"]["resolution"]


    def load_search(self, search_region):
        search_origins = []
        r_corner_1 = np.array(search_region["rectangle_c1"], dtype=np.float32)
        r_corner_2 = np.array(search_region["rectangle_c2"], dtype=np.float32)
        xy_res = search_region["resolution_xy"]
        z_res = search_region["resolution_z"]
        eps = search_region["epsilon"]
        error_value = search_region["error_value"]
        for x in np.arange(r_corner_1[0], r_corner_2[0]+xy_res, xy_res):
            for y in np.arange(r_corner_1[1], r_corner_2[1] + xy_res, xy_res):
                search_origins += [[x, y, r_corner_2[2], 0]]
        search_origins = np.array(search_origins, dtype=np.float32)

        n = (r_corner_2[2]-r_corner_1[2])/z_res
        search_params = np.zeros((), dtype=search_params_dtype)
        search_params["n"] = n
        search_params["d"] = z_res*np.array([0, 0, -1, 0], np.float32)
        search_params["eps"] = eps
        search_params["error_value"] = error_value
        return search_origins, search_params


    def load_screen(self, screen):
        center = np.array(screen["center_m"], dtype=np.float32)
        resolution = np.array(screen["resolution"], dtype=np.uint32)
        scale = np.array(screen["scale_m"], dtype=np.float32)
        rotation_vector = np.array(screen["rotation_deg"], dtype=np.float32)

        u_init = np.array([1, 0, 0], dtype=np.float32)*2*scale[0]/resolution[0]
        v_init = np.array([0, -1, 0], dtype=np.float32)*2*scale[1]/resolution[1]

        center_to_origin = np.array([-scale[0], scale[1], 0], dtype=np.float32)
        
        rotation_matrix = rotation_matrix_xyz(rotation_vector)
        u_final = rotation_matrix@u_init
        v_final = rotation_matrix@v_init
        center_to_origin = rotation_matrix@center_to_origin
        origin = center + center_to_origin
        screen_np = np.zeros((), dtype=screen_dtype)
        screen_np["u"] = pad_vector(u_final)
        screen_np["v"] = pad_vector(v_final)
        screen_np["origin"] = pad_vector(origin)
        return screen_np


    def load_camera(self, cam):
        # Computed the normal vector
        initial_z = np.array([0, 0, -1.0], dtype=np.float32)
        rotation_matrix = rotation_matrix_xyz(cam["rotation_deg"])
        normal_vector = np.array(rotation_matrix@initial_z)
        normal_vector = normal_vector/np.linalg.norm(normal_vector)

        # Units mm, must convert to meters
        focal_length = cam["focal_length_mm"]/1000

        # Units meters, as it should be
        focal_point = np.array(cam["focal_point_m"], np.float32)

        # Compute middle of camera (point on camera pixel plane)
        # This is also in millimeters. Convert to meters.
        sensor_width = cam["sensor_width_mm"]/1000
        sensor_resolution = np.array(cam["resolution"], dtype=np.uint32)
        aspect_ratio = sensor_resolution[0]/sensor_resolution[1]
        sensor_height = sensor_width/aspect_ratio
        sensor_middle = focal_point - focal_length*normal_vector
        
        # Compute the normal vectors which determine the axes of the pixels
        initial_u = np.array([-1, 0, 0], dtype=np.float32)
        initial_v = np.array([0, 1, 0], dtype=np.float32)
        rotated_u = np.array(rotation_matrix@initial_u)
        rotated_v = np.array(rotation_matrix@initial_v)
        origin = sensor_middle - rotated_u*(sensor_width/2) - rotated_v*(sensor_height/2)

        u_pixel_size = sensor_width / sensor_resolution[0]
        v_pixel_size = sensor_height / sensor_resolution[1]

        final_u = rotated_u / u_pixel_size
        final_v = rotated_v / v_pixel_size

        d = -np.dot(sensor_middle, normal_vector)

        camera_np = np.zeros((), dtype=camera_dtype)
        camera_np["focal_point"] = pad_vector(focal_point)
        camera_np["origin"]      = pad_vector(origin)
        camera_np["u"]           = pad_vector(final_u)
        camera_np["v"]           = pad_vector(final_v)
        camera_np["N"]           = pad_vector(normal_vector)
        camera_np["d"]           = d
        camera_np["res"]         = sensor_resolution
        return camera_np


