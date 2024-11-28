import random
import math
import numpy as np
import copy



def test(grid, wind_x, wind_y, wind_mag, start, goal, population_size=450, generations=22,
                                  mutation_rate=0.1, mutation_strength=6,
                                  path_length=8):
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

        # print(discriminant)

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

    def generate_random_path():
        path = []
        path.append(start)
        for _ in range(path_length):
            waypoint = [random.randint(0, grid.shape[0] - 1), random.randint(0, grid.shape[0] - 1)]
            path.append(waypoint)
        path.append(goal)
        return path

    def evaluate_fitness(path):
        d_ug = 0
        sum_W = 0
        l_traj = 0
        f_utopia = math.sqrt((goal[0] - start[0]) * (goal[0] - start[0]) + (goal[1] - start[1]) * (goal[1] - start[1]))
        steps = 0
        underground = False
        for i in range(0, len(path) - 1):
            x, y = path[i]
            next_x, next_y = path[i + 1]

            diff_x = next_x - x
            diff_y = next_y - y

            distance_P1P2 = math.sqrt(
                diff_x * diff_x +
                diff_y * diff_y
            )

            # replace 1 with resolution if necessary
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

            last_x, last_y = x, y

            for j in range(1, steps_P1P2):
                pointX = int(math.floor(x + interval_x * j))
                pointY = int(math.floor(y + interval_y * j))

                underground = pointX < 0 or pointX >= len(grid[0]) or pointY < 0 or pointY >= len(grid) or grid[pointY][
                    pointX] == 1

                if underground:
                    d_ug += step_length_P1P2
                    continue

                pointX2 = max(0, min(len(grid[0]) - 1, pointX))
                pointY2 = max(0, min(len(grid) - 1, pointY))
                # current velocity vector:
                # vc = np.array([wind_x[pointX2][pointY2], wind_y[pointX2][pointY2]])
                vc = to_vector(wind_x[pointX2][pointY2], wind_y[pointX2][pointY2], wind_mag[pointX2][pointY2])
                # unitary vector from p1 to p2
                ei = get_unit_vector(x, y, next_x, next_y)
                # nominal speed c
                c = 1

                # velocity to overcome current
                vi = c * ei - vc  #

                W = np.linalg.norm(vi) ** 2
                # print(vc, ei, pointX2, pointY2)
                c = max_speed(ei, vc, 2)

                t = travel_time(ei, vc, c, step_length_P1P2)

                # if c < 0:
                #    d_ug += step_length_P1P2

                # W

                # print(c)
                # print(vc, ei, pointX2, pointY2, W)

                sum_W += t

                # print(vc, ei, pointX2, pointY2, t)

                last_x, last_y = pointX, pointY
                steps += 1

            pointX = int(math.floor(next_x))
            pointY = int(math.floor(next_y))

            underground = pointX < 0 or pointX >= len(grid[0]) or pointY < 0 or pointY >= len(grid) or grid[pointY][
                pointX] == 1
            if underground:
                d_ug += step_length_P1P2

            # pointX2 = min(69, math.floor(pointX / 2))
            # pointY2 = min(69, math.floor(pointY / 2))
            # current velocity vector:
            #  vc = to_vector(wind_x[pointX2][pointY2], wind_y[pointX2][pointY2], wind_mag[pointX2][pointY2])
            # unitary vector from p1 to p2
            #  ei = get_unit_vector(x, y, pointX2, pointY2)
            # nominal speed c
            #  c = 2

            # velocity to overcome current
            #  vi = c * ei - vc

            #  W = np.linalg.norm(-vi) ** 2

            # sum_W += W
            # steps += 1

            last_x, last_y = pointX, pointY
        # penalty term P
        p = d_ug

        # print(steps/sum_W, steps)

        #  print(l_traj)

        #  average_energy = steps / sum_W

        # energy_utopia = 5

        # print(energy_utopia / (energy_utopia - average_energy))

        # speed = energy_utopia / (energy_utopia - average_energy)

        #  time = l_traj / speed

        # print(1 / time)

        time = sum_W
        # print((f_utopia * 5), time, (f_utopia * 5) / time)


        if p == 0.0:
            # c = speed * 0.9 + l_traj / f_utopia * 0.1
            c = time / (f_utopia * 0.3)
            return 1 + 1 / (1 + c)

        return 0 + 1 / (1 + p)

    def selection(pop, fit):
        # Implement tournament selection
        selected = []
        k = 3  # Tournament size
        for _ in range(len(pop)):
            participants = random.sample(list(zip(pop, fit)), k)
            winner = max(participants, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, path_length - 1)

        # Single-point crossover
        c1 = parent1[:crossover_point] + parent2[crossover_point:]
        c2 = parent2[:crossover_point] + parent1[crossover_point:]

        return c1, c2

    def mutate(path):
        mutated_path = path[:]
        for i in range(1, path_length - 1):
            if random.random() < mutation_rate:
                #print(mutated_path[i][0], random.randint(-mutation_strength, mutation_strength))
                mutated_path[i][0] = mutated_path[i][0] + random.randint(-mutation_strength, mutation_strength)
            if random.random() < mutation_rate:
                mutated_path[i][1] = mutated_path[i][1] + random.randint(-mutation_strength, mutation_strength)
        return mutated_path

    population = [generate_random_path() for _ in range(population_size)]

    path = [(2, 2), [7.944502423124308, 17.51801373860697], [10.138437950770838, 30.50955444162393], [10.70064531748421, 30.704623201332378], [23.363090856960362, 30.258976181022977], [26.784363936677753, 29.142703630266592], [37.11338897135945, 29.351846795139203], [50, 30], [64, 41], (68, 40)]


    fitness = evaluate_fitness(path)
    best_path = path

    print("Best fitness:", fitness)
    return best_path, [fitness]
