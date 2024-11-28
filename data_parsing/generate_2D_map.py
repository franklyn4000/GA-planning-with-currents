import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import Normalize


def generate_wind_map(file_path, grid_size, grid_points, slice_z):
    df = pd.read_csv(file_path)

    x = df["Points:0"].values
    y = df["Points:1"].values
    z = df["Points:2"].values

    values_u = df["U:0"].values
    values_v = df["U:1"].values

    # Shift x and y to start from zero
    x_min_data = np.min(x)
    y_min_data = np.min(y)

    x = x - x_min_data
    y = y - y_min_data

    x_max_data = np.max(x)
    y_max_data = np.max(y)

    # Scale x and y to range from 0 to grid_size
    x = x * (grid_size / x_max_data)
    y = y * (grid_size / y_max_data)

    x_min = 0
    x_max = grid_size
    y_min = 0
    y_max = grid_size

    x_grid, y_grid = np.mgrid[
                     x_min:x_max:grid_points * 1j,
                     y_min:y_max:grid_points * 1j
                     ]

    points = np.vstack([x, y, z]).T
    grid_points_arr = np.vstack([
        x_grid.ravel(),
        y_grid.ravel(),
        slice_z * np.ones_like(x_grid.ravel())
    ]).T

    u_grid = griddata(
        points, values_u, grid_points_arr, method='nearest'
    ).reshape(x_grid.shape)
    v_grid = griddata(
        points, values_v, grid_points_arr, method='nearest'
    ).reshape(x_grid.shape)

    magnitude_grid = np.sqrt(u_grid ** 2 + v_grid ** 2)
    magnitude_normalized = magnitude_grid / np.max(magnitude_grid)

    print(np.max(magnitude_normalized))

    return x_grid, y_grid, u_grid, v_grid, magnitude_normalized


def visualize_wind():
    x_grid, y_grid, u_grid, v_grid, magnitude_normalized = generate_wind_map(
        "../data/wind_maps0.csv", grid_size=280, grid_points=280, slice_z=5
    )

    fig, ax = plt.subplots(figsize=(12, 12))

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('jet')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ax.quiver(
        x_grid, y_grid, u_grid, v_grid,
        magnitude_normalized, cmap=cmap, norm=norm, scale=40
    )

    plt.xlim(0, 280)
    plt.ylim(0, 280)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Wind Vector Field')
    plt.show()


#visualize_wind()
