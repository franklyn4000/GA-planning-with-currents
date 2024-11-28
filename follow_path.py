import random
import math
import numpy as np
import copy
import os


def travel_time(ei, vc, c, L):
    # Compute net velocity over ground
    v_net = c * ei + vc

    # Compute the speed along the path pri
    v_path = np.dot(v_net, ei)

    if np.linalg.norm(ei) == 0:
        return 0

    if v_path <= 0:
        return 999

    # Calculate travel time
    t = L / v_path

    return t


def max_speed(ei, vc, E):
    ei_dot_vc = np.dot(ei, vc)
    vc_norm_sq = np.dot(vc, vc)

    discriminant = E + ei_dot_vc ** 2 - vc_norm_sq

    if discriminant < 0:
        return -1

    sqrt_discriminant = np.sqrt(discriminant)
    c = ei_dot_vc + sqrt_discriminant

    return c


def to_vector(dx, dy, magnitude):
    vector_x = dx * magnitude
    vector_y = dy * magnitude

    return np.array([vector_x, vector_y])


def get_unit_vector(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    if magnitude != 0:
        unit_dx = dx / magnitude
        unit_dy = dy / magnitude
    else:
        unit_dx, unit_dy = 0, 0

    return np.array([unit_dx, unit_dy])


def calculate_target_vector(heading_degrees, speed):
    heading_radians = math.radians(heading_degrees)

    x_component = speed * math.cos(heading_radians)
    y_component = speed * math.sin(heading_radians)

    return np.array([x_component, y_component])


def compute_uncertainty(point1, point2):
    scaling_factor = 0.4

    x1, y1 = point1
    x2, y2 = point2

    # Calculate Euclidean distance
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    uncertainty = scaling_factor * distance

    return uncertainty


def is_within_uncertainty(original_position, new_position, uncertainty):

    x1, y1 = original_position
    x2, y2 = new_position

    if x1 == x2 and y1 == y2:
        return True

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    print("ISWITHIN", distance, uncertainty)

    return distance <= uncertainty / 2

def simulate_target_movement(time, old_pos):

    new_pos = old_pos + calculate_target_vector(180, 1.0) * time

    return new_pos

def target_in_reach(start, goal, goal_dist):
    x1, y1 = start
    x2, y2 = goal

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance <= goal_dist


def follow(index, final_scale):
    data_folder = 'data_out'
    index_folder = f'path{index}'

    file_path = os.path.join(data_folder, index_folder)

    path = np.loadtxt(os.path.join(file_path, 'path.csv'), delimiter=',')
    grid = np.loadtxt(os.path.join(file_path, 'grid.csv'), delimiter=',')
    wind_x = np.loadtxt(os.path.join(file_path, 'u_grid.csv'), delimiter=',')
    wind_y = np.loadtxt(os.path.join(file_path, 'v_grid.csv'), delimiter=',')
    wind_mag = np.loadtxt(os.path.join(file_path, 'mag_grid.csv'), delimiter=',')
    goal_init = np.loadtxt(os.path.join(file_path, 'goal.csv'), delimiter=',')

    maxTime = 7
    time = 0

    loc = 0

    l_traj = 0

    traveled_path = []

    new_goal_pos = np.array(goal_init)
    uncertainty = 999

    while is_within_uncertainty(goal_init, new_goal_pos, uncertainty) and loc < len(path) - 1:
        time += 1



        x, y = path[loc]
        #print(x, y)
        next_x, next_y = path[loc + 1]

        diff_x = next_x - x
        diff_y = next_y - y

        distance_P1P2 = math.sqrt(
            diff_x * diff_x +
            diff_y * diff_y
        )

        steps_P1P2 = math.floor(distance_P1P2 * 1)

        if steps_P1P2 > 0:
            step_length_P1P2 = distance_P1P2 / steps_P1P2
            interval_x = diff_x / steps_P1P2
            interval_y = diff_y / steps_P1P2
        else:
            step_length_P1P2 = distance_P1P2
            interval_x = 0
            interval_y = 0

        l_traj += distance_P1P2

        for j in range(1, steps_P1P2):
            pointX = int(math.floor(x + interval_x * j))
            pointY = int(math.floor(y + interval_y * j))

            pointX2 = max(0, min(len(grid[0]) - 1, pointX))
            pointY2 = max(0, min(len(grid) - 1, pointY))

            vc = to_vector(wind_x[pointX2][pointY2], wind_y[pointX2][pointY2], wind_mag[pointX2][pointY2])
            ei = get_unit_vector(x, y, next_x, next_y)

            c = max_speed(ei, vc, 2)

            t = travel_time(ei, vc, c, step_length_P1P2)

            new_goal_pos = simulate_target_movement(t, new_goal_pos)
            uncertainty = compute_uncertainty(new_goal_pos, [pointX2, pointY2])

            #print(x, y, t, maxTime)
            traveled_path.append([pointX2, pointY2])

            maxTime -= t
            if not is_within_uncertainty(goal_init, new_goal_pos, uncertainty):
                save([pointX2, pointY2], new_goal_pos, uncertainty, index, final_scale)
                return False, traveled_path
            if target_in_reach([pointX2, pointY2], new_goal_pos, 10):
                save([pointX2, pointY2], new_goal_pos, uncertainty, index, final_scale)
                return True, traveled_path

        loc += 1

    save(path[-1], new_goal_pos, 5, index, final_scale)
    return False, traveled_path


def save(new_start, new_goal, uncertainty, index, final_scale):
    data_folder = 'data_out'
    index_folder = f'path{index + 1}'
    index_folder_prev = f'path{index}'

    file_path = os.path.join(data_folder, index_folder)
    os.makedirs(file_path, exist_ok=True)

    file_path_prev = os.path.join(data_folder, index_folder_prev)
    os.makedirs(file_path_prev, exist_ok=True)

    scaled_new_start = [new_start[0] / final_scale, new_start[1] / final_scale]
    scaled_new_goal = [new_goal[0] / final_scale, new_goal[1] / final_scale]

    np.savetxt(os.path.join(file_path, 'start.csv'), scaled_new_start, delimiter=',')
    np.savetxt(os.path.join(file_path, 'goal.csv'), scaled_new_goal, delimiter=',')
    np.savetxt(os.path.join(file_path, 'uncertainty.csv'), [uncertainty], delimiter=',')
    return False

# path = np.loadtxt('data_out/path.csv', delimiter=',')
# grid = np.loadtxt('data_out/grid.csv', delimiter=',')
# u_grid = np.loadtxt('data_out/u_grid.csv', delimiter=',')
# v_grid = np.loadtxt('data_out/v_grid.csv', delimiter=',')
# mag_grid = np.loadtxt('data_out/mag_grid.csv', delimiter=',')

# x, y = follow(grid, u_grid, v_grid, mag_grid, path)
# print(path)
# print(x, y)
