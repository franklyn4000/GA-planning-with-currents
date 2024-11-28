from generate_path import generate_path
from follow_path import follow
from visualize_path import visualize
import random
import math
import numpy as np
import copy
import os
from follow_path import compute_uncertainty

def main():
    run_demo()
    #visualize(0, [])

def run_demo():
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

    # visualize(0, [])

    # return

    reached = False
    total_traveled_path = []
    while reached is False:
        generate_path(index)
        visualize(index, total_traveled_path, True)
        reached, traveled_path = follow(index, final_scale)
        total_traveled_path.extend(traveled_path)
        print("TOTAL", total_traveled_path)
        index += 1

    # print(x, y)
    # start = [x, y]

if __name__ == "__main__":
    main()
