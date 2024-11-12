import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
import trimesh
import shapely
import networkx
import rtree
import numpy.ma as ma

# Load the wind data from the CSV
file_path = "data/wind_maps0.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Extract the points (origins of the vectors)
x = df["Points:0"].values
y = df["Points:1"].values
z = df["Points:2"].values

# Extract the velocity components
values_u = df["U:0"].values
values_v = df["U:1"].values
values_w = df["U:2"].values  # May ignore this for a horizontal slice

# Define the grid bounds based on the data
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

# Choose the z-value for the horizontal slice
slice_z = 5  # Replace with your desired z-value

# Create a 2D grid for interpolation
grid_resolution = 70j  # Adjust as needed for resolution
x_grid, y_grid = np.mgrid[x_min:x_max:grid_resolution, y_min:y_max:grid_resolution]

# Interpolate the wind components onto the grid at slice_z
points = np.vstack([x, y, z]).T
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel(), slice_z * np.ones_like(x_grid.ravel())]).T

# Use griddata to interpolate u and v onto the grid
u_grid = griddata(points, values_u, grid_points, method='nearest').reshape(x_grid.shape)
v_grid = griddata(points, values_v, grid_points, method='nearest').reshape(x_grid.shape)

# Compute the magnitude of the wind vectors
magnitude_grid = np.sqrt(u_grid**2 + v_grid**2)

# Normalize the magnitude
magnitude_normalized = magnitude_grid / np.max(magnitude_grid)

# Set up the colormap
norm = Normalize(vmin=0, vmax=1)
cmap = plt.get_cmap('jet')  # You can choose a different colormap if desired

# Create a ScalarMappable to map normalized magnitude to colors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Create a 2D plot
fig, ax = plt.subplots(figsize=(14, 12))

# Plot the wind vectors using quiver
quiver = ax.quiver(x_grid, y_grid, u_grid, v_grid,
                   magnitude_normalized, cmap=cmap, norm=norm, scale=40)

# Add a colorbar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Normalized Wind Magnitude')

occupancy_grid = np.load('occupancy_grid.npy')

# Mask the equal values
occupancy_grid_ma = ma.masked_equal(occupancy_grid, -1)

# Create a colormap where masked values will appear in black
cmap = plt.cm.gray
cmap.set_bad(color='black')


# Plot the occupancy grid
img = ax.imshow(occupancy_grid_ma, origin='lower', extent=(x_min, x_max, y_min, y_max), cmap=cmap)
cbar = fig.colorbar(img, ax=ax, label='Occupancy Value')

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title(f'Occupancy Grid at z = {slice_z}')

# Show the plot
plt.show()