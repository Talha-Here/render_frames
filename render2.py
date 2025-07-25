import pyvista as pv
import numpy as np
import os
import json

def look_at(camera_pos, target=(0, 0, 0), up=(0, 0, 1)):
    forward = np.array(target) - np.array(camera_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rot = np.stack([right, true_up, -forward], axis=1)
    trans = np.array(camera_pos).reshape(3, 1)
    mat = np.concatenate([rot, trans], axis=1)
    mat = np.vstack([mat, [0, 0, 0, 1]])
    return mat


# Set parameters
mesh_file = "Antler1F.stl"
output_dir = "antler_data2/custom_antler/train"
os.makedirs(output_dir, exist_ok=True)

# Load and normalize mesh
mesh = pv.read(mesh_file)
mesh.compute_normals(inplace=True)
mesh.translate(-np.array(mesh.center), inplace=True)
mesh.scale(1.0 / max(mesh.length, 1e-8), inplace=True)

# Increase resolution
plotter = pv.Plotter(off_screen=True, window_size=(1024, 1024))  # RES
plotter.set_background("black")
plotter.add_mesh(mesh, color="wheat", smooth_shading=True)
plotter.enable_lightkit()

# Camera settings
n_frames = 60
radius = 2.5
elevation = 0.4
fov_deg = 40
fov_rad = np.radians(fov_deg)

transforms = {
    "camera_angle_x": fov_rad,
    "frames": []
}

# Render loop
for i in range(n_frames):
    angle = 2 * np.pi * i / n_frames
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = elevation
    camera_pos = (x, y, z)

    plotter.camera_position = [camera_pos, (0, 0, 0), (0, 0, 1)]

    filename = f"img_{i:03d}.png"
    img_path = os.path.join(output_dir, filename)
    plotter.show(auto_close=False)
    plotter.screenshot(img_path)

    transform_matrix = look_at(camera_pos).tolist()
    transforms["frames"].append({
        "file_path": f"./train/img_{i:03d}",
        "transform_matrix": transform_matrix
    })

# Save transform
json_path = os.path.join("antler_data2/custom_antler", "transforms_train.json")
with open(json_path, "w") as f:
    json.dump(transforms, f, indent=4)

print("images and transforms_train.json generated.")
