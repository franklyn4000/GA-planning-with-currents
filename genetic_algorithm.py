import random
import math
import numpy as np


def genetic_algorithm_pathfinding(grid, wind_x, wind_y, wind_mag, start, goal, population_size=1600, generations=90,
                                  mutation_rate=0.15, mutation_strength=10,
                                  path_length=6):
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
            waypoint = (random.randint(0, grid.shape[0] - 1), random.randint(0, grid.shape[0] - 1))
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

                pointX2 = min(34, pointX)
                pointY2 = min(34, pointY)
                # current velocity vector:
                vc = np.array([wind_x[pointX2][pointY2], wind_y[pointX2][pointY2]])
                # vc = [to_vector(], , wind_mag[pointX2][pointY2])
                # unitary vector from p1 to p2
                ei = get_unit_vector(last_x, last_y, pointX2, pointY2)
                # nominal speed c
                c = 1

                # velocity to overcome current
                vi = c * ei - vc

                W = np.linalg.norm(vi) ** 2

               # print(vc, ei, pointX2, pointY2, W)

                sum_W += W

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

        average_energy = steps / sum_W

        #print(average_energy)

        energy_utopia = 0.1

        if p == 0.0:
            c = energy_utopia / average_energy * 0.9 + l_traj / f_utopia * 0.1
            return 1 + 1 / (1 + c)

        return 0 + 1 / (1 + p)

    def selection(pop, fit):
        selected = []
        for _ in range(len(pop)):
            i1, i2 = random.sample(range(len(pop)), 2)
            if fit[i1] > fit[i2]:
                selected.append(pop[i1])
            else:
                selected.append(pop[i2])
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
                mutated_path[i] = (mutated_path[i][0] + random.randint(-mutation_strength, mutation_strength),
                                   mutated_path[i][0] + random.randint(-mutation_strength, mutation_strength))
        return mutated_path

    population = []

    for _ in range(population_size):
        path = generate_random_path()
        population.append(path)

    best_path = None
    best_fitness = float('-inf')
    fitnesses = []
    best_fitnesses = []

    for generation in range(generations):
        print("generation:", generation)

        if best_path is not None:
            population = selection(population, fitnesses)
            new_population = []
            for i in range(0, len(population), 2):
                parent1 = population[i]
                if i + 1 < len(population):
                    parent2 = population[i + 1]
                else:
                    parent2 = population[0]
                child1, child2 = crossover(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)
            population = [mutate(path) for path in new_population]

        # Evaluate fitness
        fitnesses = []
        for path in population:
            fitness = evaluate_fitness(path)
            fitnesses.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_path = path[:]
        best_fitnesses.append(best_fitness)

    print("best fitness: ", best_fitness)

    # print(population)
    return best_path, best_fitnesses
