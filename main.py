from genetic_algorithm import computePath
import json

def read_grid(filename):
    """Reads an occupancy grid from a file."""
    with open(filename, 'r') as file:
        grid = []
        for line in file:
            grid.append([int(cell) for cell in line.split()])
    return grid

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
    grid = read_grid("./grid.json")

    computePath(grid, start, goal)


if __name__ == "__main__":
    main()
