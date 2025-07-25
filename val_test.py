import pyvista as pv
import numpy as np
import os
import json

def look_at(camera_pos, target=(0, 0, 0), up=(0, 0, 1)):
    forward = np.array(target) - np.array(camera_pos)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rot = np.stack([right, true_up, -forward], axis=1)
    trans = np.array(camera_pos).reshape(3, 1)
    mat = np.concatenate([rot, trans], axis=1)
    mat = np.vstack([mat, [0, 0, 0, 1]])
    return mat


mesh_file = "Antler1F.stl"                 
output_base = "antler_data2/custom_antler/test"     

mode = "test"   # val or test
n_frames = 30  # fewer for val/test
radius = 2.5   # distance from center
elevation = 0.7 if mode == "val" else 0.3   # height of camera
fov_deg = 40   # same as train
image_res = (1024, 1024)  # must match training


output_dir = os.path.join(output_base, mode)
os.makedirs(output_dir, exist_ok=True)

# Load and normalize the mesh
mesh = pv.read(mesh_file)
mesh.compute_normals(inplace=True)
mesh.translate(-np.array(mesh.center), inplace=True)
mesh.scale(1.0 / max(mesh.length, 1e-8), inplace=True)

plotter = pv.Plotter(off_screen=True, window_size=image_res)
plotter.set_background("black")
plotter.add_mesh(mesh, color="wheat", smooth_shading=True)
plotter.enable_lightkit()

fov_rad = np.radians(fov_deg)
transforms = {
    "camera_angle_x": fov_rad,
    "frames": []
}

for i in range(n_frames):
    angle = 2 * np.pi * i / n_frames
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = elevation
    cam_pos = (x, y, z)

    plotter.camera_position = [cam_pos, (0, 0, 0), (0, 0, 1)]
    img_name = f"img_{i:03d}.png"
    img_path = os.path.join(output_dir, img_name)
    plotter.show(auto_close=False)
    plotter.screenshot(img_path)

    transform_matrix = look_at(cam_pos).tolist()
    transforms["frames"].append({
        "file_path": f"./{mode}/img_{i:03d}",
        "transform_matrix": transform_matrix
    })

json_path = os.path.join(output_base, f"transforms_{mode}.json")
with open(json_path, "w") as f:
    json.dump(transforms, f, indent=4)

print(f"Done: {mode} images and transforms saved to {output_dir}")
