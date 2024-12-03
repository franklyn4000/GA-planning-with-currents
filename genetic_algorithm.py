import random
import math
import numpy as np
import copy


def genetic_algorithm_pathfinding(grid, wind_x, wind_y, wind_mag, start, goal, population_size, generations,
                                  initial_mutation_rate, initial_mutation_strength,
                                  path_length, init_pop=None):
    def travel_time(c, L):

        if c <= 0:
            return 999

        if L == 0:
            return 0

        # Calculate travel time
        t = L / c

        return t

    def max_speed(ei, vc, E):
        ei_dot_vc = np.dot(ei, vc)
        vc_norm_sq = np.dot(vc, vc)

        a = 1
        b = -2 * ei_dot_vc
        c_coef = vc_norm_sq - E

        discriminant = b ** 2 - 4 * a * c_coef

        if discriminant < 0:
            return -1

        sqrt_discriminant = np.sqrt(discriminant)
        c = (-b + sqrt_discriminant) / (2 * a)

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

                vc = to_vector(wind_x[pointX2][pointY2], wind_y[pointX2][pointY2], wind_mag[pointX2][pointY2])
                ei = get_unit_vector(x, y, next_x, next_y)

                c = max_speed(ei, vc, 2)

                t = travel_time(c, step_length_P1P2)

                sum_W += t

                steps += 1

            pointX = int(math.floor(next_x))
            pointY = int(math.floor(next_y))

            underground = pointX < 0 or pointX >= len(grid[0]) or pointY < 0 or pointY >= len(grid) or grid[pointY][
                pointX] == 1
            if underground:
                d_ug += step_length_P1P2

        p = d_ug

        time = sum_W

        if p == 0.0:
            c = time / (f_utopia * 0.25)
            return 1 + 1 / (1 + c)

        return 0 + 1 / (1 + p)

    def selection(pop, fit):
        # Implement tournament selection
        selected = []
        k = 3
        for _ in range(len(pop)):
            participants = random.sample(list(zip(pop, fit)), k)
            winner = max(participants, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, path_length - 1)

        c1 = parent1[:crossover_point] + parent2[crossover_point:]
        c2 = parent2[:crossover_point] + parent1[crossover_point:]

        return c1, c2

    def mutate(path, rate, strength):
        mutated_path = copy.deepcopy(path)
        for i in range(1, path_length + 1):
            if random.random() < rate:
                mutated_path[i][0] += random.gauss(0, mutation_strength)
                mutated_path[i][1] += random.gauss(0, mutation_strength)
        return mutated_path

    if init_pop is not None:
        population = init_pop
    else:
        population = [generate_random_path() for _ in range(population_size)]

    best_path = None
    best_fitnesses = []
    elitism_rate = 0.1
    elite_size = max(1, int(elitism_rate * population_size))

    mutation_rate = initial_mutation_rate
    max_mutation_rate = 0.1
    stagnation_threshold = 7
    stagnation_counter = 0

    mutation_strength = initial_mutation_strength
    strength_stagnation_counter = 0

    best_fitness = -1
    best_fitness_run = []
    average_fitness_run = []

    for generation in range(generations):
        if init_pop is not None:
            print(f"Generation: {generation} mutation rate: {mutation_rate} mutation strength: {mutation_strength}")
        fitnesses = [evaluate_fitness(path) for path in population]
        current_best_fitness = max(fitnesses)
        best_index = fitnesses.index(current_best_fitness)
        best_path = population[best_index]
        best_fitnesses.append(current_best_fitness)

        if best_fitness is None or current_best_fitness > best_fitness:
            # Improvement found
            best_fitness = current_best_fitness
            stagnation_counter = 0
            strength_stagnation_counter = 0
            mutation_rate = initial_mutation_rate
            mutation_strength = initial_mutation_strength
        else:
            # No improvement
            stagnation_counter += 1
            if stagnation_counter >= stagnation_threshold:
                mutation_rate = min(mutation_rate * 1.5, max_mutation_rate)
                stagnation_counter = 0  # Reset counter after adjusting mutation rate
                if mutation_rate == max_mutation_rate:
                    strength_stagnation_counter += 1
                    if strength_stagnation_counter > 10:
                        break

        avg_fitness = sum(fitnesses) / len(fitnesses)

        average_fitness_run.append(avg_fitness)
        best_fitness_run.append(current_best_fitness)

        # Sort population by fitness
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
        # Deep copy elites
        new_population = [copy.deepcopy(individual) for individual in sorted_population[:elite_size]]

        remaining_population = sorted_population[elite_size:]
        remaining_fitnesses = sorted(fitnesses, reverse=True)[elite_size:]

        if remaining_population:
            selected = selection(remaining_population, remaining_fitnesses)
        else:
            selected = []

        # Generate new offspring
        while len(new_population) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, mutation_strength)
            child2 = mutate(child2, mutation_rate, mutation_strength)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    print("Best fitness:", best_fitness)
    return best_path, best_fitnesses, best_fitness_run, average_fitness_run
