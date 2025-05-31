import pyvista as pv
import pandas as pd
import sys
import numpy as np

# Load data
csv_file = sys.argv[1] if len(sys.argv) > 1 else "kick_simulation_data.csv"
df = pd.read_csv(csv_file)

# Initialize PyVista plotter
plotter = pv.Plotter(window_size=[1600, 900])  # Large 16:9 window
plotter.set_background("white")

# Bounding box for the free kick area
x_min, x_max = -5, 35
z_min, z_max = 24, 44
y_min, y_max = 0, 10

# Add ground plane
ground = pv.Plane(
    center=((x_min + x_max) / 2, 0.0, (z_min + z_max) / 2),
    direction=(0, 1, 0),
    i_size=(x_max - x_min),
    j_size=(z_max - z_min)
)
plotter.add_mesh(ground, color="lightgreen", opacity=0.3)

# Helper: Create rotated box (wall/goal) from normal vector (orientation)
def add_oriented_box(center, width, height, normal, color, opacity=0.5, thickness=0.3):
    normal = np.array(normal)
    if np.linalg.norm(normal) < 1e-6:
        normal = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)

    up = np.array([0, 1, 0])

    if abs(np.dot(normal, up)) > 0.99:
        up = np.array([0, 0, 1])

    width_dir = np.cross(up, normal)
    width_dir /= np.linalg.norm(width_dir)

    height_dir = np.cross(normal, width_dir)
    height_dir /= np.linalg.norm(height_dir)

    transform = np.eye(4)
    transform[:3, 0] = width_dir     # X (width)
    transform[:3, 1] = height_dir    # Y (height)
    transform[:3, 2] = normal        # Z (thickness/normal)
    transform[:3, 3] = center        # position

    box = pv.Cube(center=(0, 0, 0), x_length=width, y_length=height, z_length=thickness)
    box.transform(transform)

    plotter.add_mesh(box, color=color, opacity=opacity)

# Add walls and goals with corrected orientation
for _, row in df.iterrows():
    center = np.array([row["x"], row["y"], row["z"]])
    width = row["width"]
    height = row["height"]
    normal = np.array([row["orientation_x"], row["orientation_y"], row["orientation_z"]])

    corrected_center = center.copy()
    corrected_center[1] += height / 2.0  # Center adjustment

    if row["object_type"] == "wall":
        add_oriented_box(corrected_center, width, height, normal, color="blue", opacity=0.5)
    elif row["object_type"] == "goal":
        add_oriented_box(corrected_center, width, height, normal, color="green", opacity=0.5)

# Ball trajectory
ball_points = df[df["object_type"] == "ball"][["x", "y", "z"]].values
if len(ball_points) >= 2:
    ball_path = pv.Spline(ball_points, 1000)
    plotter.add_mesh(ball_path, color="red", line_width=3)

for point in ball_points[::10]:
    sphere = pv.Sphere(radius=0.11, center=point)
    plotter.add_mesh(sphere, color="red", opacity=0.8)

# Camera settings - behind ball, elevated, aligned with field center
field_width = 68.0  # meters

goal_center = np.array([0, 1.0, field_width / 2.0])  # Focus on goal center (0,1,34)
camera_pos = np.array([120.0, 30.0, field_width / 2.0])  # Behind ball, elevated

plotter.camera_position = [camera_pos, goal_center, (0, 1, 0)]

# Axes and grid
plotter.add_axes()
plotter.show_grid(xlabel='X (meters)', ylabel='Y (meters)', zlabel='Z (meters)')

# Show plot
plotter.show()
