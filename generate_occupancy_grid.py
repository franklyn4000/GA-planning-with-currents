import numpy as np
import matplotlib.pyplot as plt
import trimesh
import matplotlib.path as mplPath
import numpy.ma as ma
from shapely.affinity import translate
# Load the STL mesh
mesh = trimesh.load('data/outer_environment.stl')

# Apply any necessary transformations to align the mesh with your coordinate system
# For example, scaling, rotation, or translation
# Uncomment and adjust the following transformations if needed:

# Scaling example (e.g., from millimeters to meters)
# mesh.apply_scale(0.001)

# Rotation example (rotate 180 degrees around the Z-axis)
# angle = np.pi  # 180 degrees in radians
# rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
# mesh.apply_transform(rotation_matrix)

# Translation example
# Since the mesh is centered at (0, 0), we can translate it by half of its dimensions in the positive x and y directions.

# Calculate half of the mesh's dimensions
mesh_bounds = mesh.bounds
x_min, y_min = mesh_bounds[0][0], mesh_bounds[0][1]
x_max, y_max = mesh_bounds[1][0], mesh_bounds[1][1]
x_half = (x_max - x_min) / 2
y_half = (y_max - y_min) / 2

x_padding = -40
y_padding = 0

# Apply translation by half in x and y directions
translation_vector = [x_half + x_padding, -y_half + y_padding, 0]
#mesh.apply_translation(translation_vector)

# Update mesh bounds after translation
mesh_bounds = mesh.bounds
x_min, y_min = mesh_bounds[0][0], mesh_bounds[0][1]
x_max, y_max = mesh_bounds[1][0], mesh_bounds[1][1]

# Choose the z-value for the horizontal slice
slice_z = 5  # Replace with your desired z-value

# Slice the mesh at z = slice_z
plane_origin = [0, 0, slice_z]
plane_normal = [0, 0, 1]

# Get the intersection of the mesh with the plane
slice = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

if slice is not None:
    # Convert the slice to 2D paths
    slice_2D, _ = slice.to_planar()
    # Get the polygons representing the cross-section at slice_z
    polygons = slice_2D.polygons_full
else:
    print(f"No intersection found at z = {slice_z}")
    polygons = []

# Define the grid bounds based on the mesh bounds
# Since we've updated the mesh bounds after translation, we use them here
x_min, y_min = mesh_bounds[0][0], mesh_bounds[0][1]
x_max, y_max = mesh_bounds[1][0], mesh_bounds[1][1]

# Define the grid resolution
grid_resolution = 140  # Adjust as needed for finer or coarser grid

# Create a 2D grid over the x and y domain
x_grid = np.linspace(-35, 35, grid_resolution)
y_grid = np.linspace(-35, 35, grid_resolution)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Flatten the grid points into a list of (x, y) points
grid_points_2d = np.vstack((X_grid.ravel(), Y_grid.ravel())).T

# Initialize the inside_mask array
inside_mask = np.zeros(grid_points_2d.shape[0], dtype=bool)

# Replace with your desired translation values
translation_vector = (4, -2)

# For each polygon, check which grid points are inside
for polygon in polygons:
    translated_polygon = translate(polygon, xoff=translation_vector[0], yoff=translation_vector[1])

    x_poly, y_poly = translated_polygon.exterior.xy

    # Create a path from the translated polygon's exterior coordinates
    poly_path = mplPath.Path(np.vstack((x_poly, y_poly)).T)

    # Determine which grid points are inside the translated polygon
    inside = poly_path.contains_points(grid_points_2d)

    # Update the mask to include points inside this translated polygon
    inside_mask = inside_mask | inside  # Logical OR to accumulate

# Reshape inside_mask to the shape of the grid
inside_mask = inside_mask.reshape(X_grid.shape)

# Create the occupancy grid
occupancy_grid = np.zeros(X_grid.shape, dtype=float)

# Set values to -1 where points are inside the mesh
occupancy_grid[inside_mask] = 1

# Save the occupancy grid to a file
np.save('occupancy_grid.npy', occupancy_grid)
print("Occupancy grid saved to 'occupancy_grid.npy'")

# Load the occupancy grid from the file
loaded_occupancy_grid = np.load('occupancy_grid.npy')
print("Occupancy grid loaded from 'occupancy_grid.npy'")

# Mask the -1 values for visualization
occupancy_grid_ma = ma.masked_equal(occupancy_grid, 1)

# Create a colormap where masked values will appear in black
cmap = plt.cm.gray
cmap.set_bad(color='black')

# Plot the occupancy grid
plt.figure(figsize=(10, 8))
plt.imshow(occupancy_grid_ma, origin='lower', extent=(0, grid_resolution, 0, grid_resolution), cmap=cmap)
plt.colorbar(label='Occupancy Value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Occupancy Grid at z = {slice_z}')
plt.show()


print("STL Mesh Coordinate Ranges:")
print(f"X: {mesh.bounds[0][0]} to {mesh.bounds[1][0]}")
print(f"Y: {mesh.bounds[0][1]} to {mesh.bounds[1][1]}")
print(f"Z: {mesh.bounds[0][2]} to {mesh.bounds[1][2]}")