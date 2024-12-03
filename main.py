from generate_path import generate_path
from follow_path import follow
from visualize_path import visualize
import numpy as np
import os
from follow_path import compute_uncertainty
import json

import matplotlib.pyplot as plt

def main():
    run_demo()
    #run_single()
    #visualize(0, [])


def run_single():
    start = [2, 2]
    goal = [26, 32]
    index = 0
    final_scale = 4

    data_folder = 'data_out'
    index_folder = f'path{index}'

    file_path = os.path.join(data_folder, index_folder)
    os.makedirs(file_path, exist_ok=True)

    np.savetxt(os.path.join(file_path, 'start.csv'), start, delimiter=',')
    np.savetxt(os.path.join(file_path, 'goal.csv'), goal, delimiter=',')
    np.savetxt(os.path.join(file_path, 'uncertainty.csv'), [
        compute_uncertainty([start[0] * final_scale, start[1] * final_scale],
                            [goal[0] * final_scale, goal[1] * final_scale])], delimiter=',')

    generate_path(index)
    fit_evo = np.loadtxt(os.path.join(file_path, 'fit_evo.csv'), delimiter=',')
    return fit_evo
    visualize(index, [], True)

def read_config():
    try:
        with open('config.json', 'r') as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        print(f"Error: The file config.json was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file config.json is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def run_demo():

    config = read_config()
    index = 0


    data_folder = 'data_out'
    index_folder = f'path{index}'

    file_path = os.path.join(data_folder, index_folder)
    os.makedirs(file_path, exist_ok=True)

    np.savetxt(os.path.join(file_path, 'start.csv'), config["start"], delimiter=',')
    np.savetxt(os.path.join(file_path, 'goal.csv'), config["goal"], delimiter=',')
    np.savetxt(os.path.join(file_path, 'uncertainty.csv'), [
        compute_uncertainty([config["start"][0] * config["final_scale"], config["start"][1] * config["final_scale"]],
                            [config["goal"][0] * config["final_scale"], config["goal"][1] * config["final_scale"]])], delimiter=',')



    reached = False
    total_traveled_path = []
    while reached is False:
        generate_path(index, config)
        visualize(index, total_traveled_path, True)
        reached, traveled_path = follow(index, config)
        total_traveled_path.extend(traveled_path)
        index += 1

if __name__ == "__main__":
    main()
