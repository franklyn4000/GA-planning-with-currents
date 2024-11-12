from genetic_algorithm import genetic_algorithm_pathfinding
import json
import numpy as np
import matplotlib.pyplot as plt

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



def main():
    import sys

    start, goal = read_config("./config.json")
    grid = read_grid("occupancy_grid.npy")

    path, fit_evo = genetic_algorithm_pathfinding(grid, start, goal)

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

        # Add a legend
        ax.legend()
    else:
        print("No path found")

    plt.show()

if __name__ == "__main__":
    main()
