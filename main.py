from genetic_algorithm import genetic_algorithm_pathfinding
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata

def read_grid(filename):
    occupancy_grid = np.load(filename)
    return occupancy_grid

def read_config(config_file):
    """Reads the start and goal coordinates from a JSON config file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    start = tuple(config["start"])
    goal = tuple(config["goal"])
    return start, goal


def read_wind(filename):
    df = pd.read_csv(filename)

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
    magnitude_grid = np.sqrt(u_grid ** 2 + v_grid ** 2)

    # Normalize the magnitude
    magnitude_normalized = magnitude_grid / np.max(magnitude_grid)

    return u_grid, v_grid, magnitude_normalized


def main():
    import sys

    start, goal = read_config("./config.json")
    grid = read_grid("occupancy_grid.npy")
    wind_x, wind_y, wind_mag = read_wind("data/wind_maps0.csv")

    path, fit_evo = genetic_algorithm_pathfinding(grid, wind_x, wind_y, wind_mag, start, goal)

   # print(path)
    plt.plot(fit_evo, marker='o', linestyle='-', color='b')

    # Adding title and labels
    plt.title('Series of Float Numbers')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Displaying the plot
    plt.show()

    fig, ax = plt.subplots(figsize=(14, 12))
    nrows, ncols = grid.shape

    resolution = 140

    # Plot the occupancy grid
    img = ax.imshow(grid, origin='lower', extent=(0, resolution, 0, resolution))
    cbar = fig.colorbar(img, ax=ax, label='Occupancy Value')

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, resolution)
    ax.set_ylim(0, resolution)
    ax.set_title(f'Occupancy Grid at z = {5}')

    if path is not None:
        # Extract x and y coordinates from the path
        x_coords = [pos[0] * (resolution / ncols) for pos in path]
        y_coords = [pos[1] * (resolution / nrows) for pos in path]

        # Plot the path
        ax.plot(x_coords, y_coords, color='red', linewidth=2, label='Path')

        # Plot the start and goal points
        start_x = start[0] * (resolution / ncols)
        start_y = start[1] * (resolution / nrows)
        goal_x = goal[0] * (resolution / ncols)
        goal_y = goal[1] * (resolution / nrows)

        ax.plot(start_x, start_y, marker='o', color='green', markersize=10, label='Start')
        ax.plot(goal_x, goal_y, marker='o', color='blue', markersize=10, label='Goal')

    else:
        print("No path found")

    plt.show()

if __name__ == "__main__":
    main()
