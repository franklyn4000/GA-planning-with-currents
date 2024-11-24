import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import Normalize

def generate_wind_map(file_path, grid_resolution, slice_z):
    df = pd.read_csv(file_path)
    grid_resolution = grid_resolution * 1j

    x = df["Points:0"].values
    y = df["Points:1"].values
    z = df["Points:2"].values

    values_u = df["U:0"].values
    values_v = df["U:1"].values

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_grid, y_grid = np.mgrid[x_min:x_max:grid_resolution, y_min:y_max:grid_resolution]

    points = np.vstack([x, y, z]).T
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel(), slice_z * np.ones_like(x_grid.ravel())]).T

    u_grid = griddata(points, values_u, grid_points, method='nearest').reshape(x_grid.shape)
    v_grid = griddata(points, values_v, grid_points, method='nearest').reshape(x_grid.shape)

    magnitude_grid = np.sqrt(u_grid**2 + v_grid**2)
    magnitude_normalized = magnitude_grid / np.max(magnitude_grid)

    return x_grid, y_grid, u_grid, v_grid, magnitude_normalized


x_grid, y_grid, u_grid, v_grid, magnitude_normalized = generate_wind_map("data/wind_maps0.csv", 35, 5)

fig, ax = plt.subplots(figsize=(12, 12))

norm = Normalize(vmin=0, vmax=1)
cmap = plt.get_cmap('jet')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

ax.quiver(x_grid, y_grid, u_grid, v_grid,
                   magnitude_normalized, cmap=cmap, norm=norm, scale=40)