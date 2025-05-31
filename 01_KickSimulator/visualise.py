import pyvista as pv
import pandas as pd

# Load data
csv_file = "output/wavepoints.csv"
df = pd.read_csv(csv_file)

# Initialize PyVista plotter
plotter = pv.Plotter()
plotter.set_background("white")

# Define the bounding box for the area of interest
# For example, 40m x 20m x 20m around the free kick area (x: 0 to 40, z: center ±10)
x_min, x_max = -5, 35   # 5m behind goal, 35m in front
z_min, z_max = 24, 44   # 10m left/right from center
y_min, y_max = 0, 10    # Ground to ~10m height

# Add ground plane (just a small patch)
ground = pv.Plane(
    center=((x_min + x_max) / 2, 0.0, (z_min + z_max) / 2),
    direction=(0, 1, 0),
    i_size=(x_max - x_min),
    j_size=(z_max - z_min)
)
plotter.add_mesh(ground, color="lightgreen", opacity=0.3)

# Add wall and goal
for idx, row in df.iterrows():
    center = [row["x"], row["y"], row["z"]]
    width = row["width"]
    height = row["length"]

    if row["object_type"] == "wall":
        wall = pv.Cube(center=center, x_length=0.3, y_length=height, z_length=width)
        plotter.add_mesh(wall, color="blue", opacity=0.5)
    elif row["object_type"] == "goal":
        goal = pv.Cube(center=center, x_length=0.5, y_length=height, z_length=width)
        plotter.add_mesh(goal, color="green", opacity=0.5)

# Ball trajectory
ball_points = df[df["object_type"] == "ball"][["x", "y", "z"]].values
ball_path = pv.Spline(ball_points, 1000)
plotter.add_mesh(ball_path, color="red", line_width=3)

# Ball spheres (sampled)
for point in ball_points[::10]:
    sphere = pv.Sphere(radius=0.11, center=point)
    plotter.add_mesh(sphere, color="red", opacity=0.8)

# Camera settings
plotter.camera_position = [
    ((x_max + x_min) / 2, (y_max - y_min), (z_max + z_min) / 2 + 10),
    ((x_max + x_min) / 2, 0, (z_max + z_min) / 2),
    (0, 1, 0)
]

plotter.set_focus((x_max / 2, 1.0, (z_min + z_max) / 2))

# Axes and grid
plotter.add_axes()
plotter.show_grid(xlabel='X (meters)', ylabel='Y (meters)', zlabel='Z (meters)')

# Show plot
plotter.show()
