from genetic_algorithm import genetic_algorithm_pathfinding
import numpy as np
from data_parsing.generate_occupancy_grid import generate_grid
from data_parsing.generate_2D_map import generate_wind_map
import copy
import os
import math

def insert_waypoints(waypoints, goal, N):
    new_waypoints = []
    num_points = len(waypoints)
    for i in range(num_points - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        # Add the start point
        new_waypoints.append(start)
        # Compute and add the intermediate waypoints
        for j in range(1, N + 1):
            t = j / (N + 1)
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            new_waypoints.append([x, y])
    # Add the last waypoint
    new_waypoints.append(goal)
    return new_waypoints


def repeat_paths(paths, times):
    new_paths = []
    for path in paths:
        for _ in range(times):
            # Use deepcopy to ensure each path is a separate object
            new_paths.append(copy.deepcopy(path))
    return new_paths


def scale_path(path, N):
    scaled_path = [[x * N, y * N] for [x, y] in path]
    return scaled_path


def map_distance_to_values(distance):
    x = 3 + (distance - 10) * 1 / 150

    y = 0 + (distance - 10) * 3 / 150


    x = int(round(x))
    y = int(round(y))

    print("COMPUTED", distance, x, y)

    return [x, y]


def generate_path(index):
    data_folder = 'data_out'
    index_folder = f'path{index}'

    file_path = os.path.join(data_folder, index_folder)
    os.makedirs(file_path, exist_ok=True)

    start = np.loadtxt(os.path.join(file_path, 'start.csv'), delimiter=',')
    goal = np.loadtxt(os.path.join(file_path, 'goal.csv'), delimiter=',')
    #print("STAR", start)




    slice = 5
    final_population = []

    desired_population = 8
    init_pop = 40
    init_gen = 45
    init_mut_rate = 0.03
    init_mut_strength = 2


    final_scale = 4
    final_pop = desired_population
    final_gen = 100
    final_mut_rate = 0.005
    final_mut_strength = 1

    res = 35

    start_final = [start[0] * final_scale, start[1] * final_scale]
    goal_final = [goal[0] * final_scale, goal[1] * final_scale]
    res_final = res * final_scale
    grid = generate_grid(res, slice, 'data/outer_environment.stl')
    x_grid, y_grid, u_grid, v_grid, mag_grid = generate_wind_map("data/wind_maps0.csv", res, res, slice)

    x1, y1 = start_final
    x2, y2 = goal_final
    distance_start_goal = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    a, b = map_distance_to_values(distance_start_goal)

    init_path_length = a
    waypoints_to_insert = b

    # generate <desired_population> low_res parts as initial population for the final GA
    for i in range(desired_population):
        path, fit_evo = genetic_algorithm_pathfinding(grid, u_grid, v_grid, mag_grid, start, goal, init_pop, init_gen,
                                                      init_mut_rate, init_mut_strength,
                                                      init_path_length, None)
        scaled_path = scale_path(path, final_scale)
        new_path = insert_waypoints(scaled_path, goal_final, waypoints_to_insert)
        final_population.append(new_path)

    final_path_length = len(final_population[0]) - 2

    final_population_repeated = repeat_paths(final_population, 5)

    grid_final = generate_grid(res_final, slice, 'data/outer_environment.stl')
    x_grid_final, y_grid_final, u_grid_final, v_grid_final, mag_grid_final = generate_wind_map("data/wind_maps0.csv",
                                                                                               res_final, res_final,
                                                                                               slice)

    path, fit_evo = genetic_algorithm_pathfinding(grid_final, u_grid_final, v_grid_final, mag_grid_final, start_final,
                                                  goal_final, final_pop, final_gen,
                                                  final_mut_rate, final_mut_strength,
                                                  final_path_length, final_population_repeated)

    np_path = np.array(path)



    np.savetxt(os.path.join(file_path, 'path.csv'), np_path, delimiter=',')
    np.savetxt(os.path.join(file_path, 'fit_evo.csv'), fit_evo, delimiter=',')
    np.savetxt(os.path.join(file_path, 'res.csv'), [res_final], delimiter=',')
    np.savetxt(os.path.join(file_path, 'grid.csv'), grid_final, delimiter=',')
    np.savetxt(os.path.join(file_path, 'x_grid.csv'), x_grid_final, delimiter=',')
    np.savetxt(os.path.join(file_path, 'y_grid.csv'), y_grid_final, delimiter=',')
    np.savetxt(os.path.join(file_path, 'u_grid.csv'), u_grid_final, delimiter=',')
    np.savetxt(os.path.join(file_path, 'v_grid.csv'), v_grid_final, delimiter=',')
    np.savetxt(os.path.join(file_path, 'mag_grid.csv'), mag_grid_final, delimiter=',')
    np.savetxt(os.path.join(file_path, 'start.csv'), start_final, delimiter=',')
    np.savetxt(os.path.join(file_path, 'goal.csv'), goal_final, delimiter=',')