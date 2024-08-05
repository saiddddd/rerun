import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(42)

def long_tail_opposite_learning(position, lowerbound, upperbound, distribution_type):
    if distribution_type == 'pareto':
        shape = 2.0  # Shape parameter for Pareto distribution
        scale = 1.0  # Scale parameter for Pareto distribution
        opposite_position = upperbound - np.abs(position * (np.random.pareto(shape) + 1) * scale - lowerbound)
    elif distribution_type == 'weibull':
        shape = 1.5  # Shape parameter untuk Weibull distribution
        scale = 1.0  # Scale parameter untuk Weibull distribution
        opposite_position = upperbound - np.abs(position * np.random.weibull(shape) * scale - lowerbound)
    elif distribution_type == 'cauchy':
        scale = 1.0  # Scale parameter untuk Cauchy distribution
        opposite_position = upperbound - np.abs(position * np.random.standard_cauchy() * scale - lowerbound)
    elif distribution_type == 'lomax':
        shape = 2.0  # Shape parameter untuk Lomax distribution
        scale = 1.0  # Scale parameter untuk Lomax distribution
        opposite_position = upperbound - np.abs(position * (np.random.pareto(shape) + 1) * scale - lowerbound)
    elif distribution_type == 'lognormal':
        mean = 0.0  # Mean parameter untuk Log-normal distribution
        sigma = 1.0  # Sigma parameter untuk Log-normal distribution
        opposite_position = upperbound - np.abs(position * np.random.lognormal(mean, sigma) - lowerbound)
    else:
        opposite_position = position
    return opposite_position

def initialize_wolves(num_wolves, num_dimensions, search_space):
    return [np.random.uniform(search_space[0], search_space[1], num_dimensions) for _ in range(num_wolves)]

def adaptive_parameter(t, max_iterations, initial_value, final_value):
    return initial_value * ((final_value / initial_value) ** (t / max_iterations))

def MGWO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    # Initialize a population of wolves
    wolves = initialize_wolves(SearchAgents, dimension, (lowerbound, upperbound))

    # Initialize Alpha, Beta, and Delta wolves
    alpha, beta, delta = wolves[0], wolves[1], wolves[2]
    alpha_fitness, beta_fitness, delta_fitness = fitness(alpha), fitness(beta), fitness(delta)

    trajectories = []  # To store the trajectory of solutions
    fitness_history = []  # To store fitness history
    position_history = []  # To store position history
    Alpha_pos_history = []  # To store history of Alpha position

    Exploration = np.zeros(Max_iterations)
    Exploitation = np.zeros(Max_iterations)

    # Define initial and final mutation rates for adaptive mutation
    initial_mutation_rate = 0.2
    final_mutation_rate = 0.01

    # Define OLB parameters
    olb_probability = 0.2  # Probability of applying OLB
    distribution_types = ['pareto', 'weibull', 'cauchy', 'lomax', 'lognormal']
    current_distribution = distribution_types[0]
    no_improvement_counter = 0

    print("MGWO is optimizing  \"" + fitness.__name__ + "\"")

    for t in range(Max_iterations):
        # Calculate adaptive parameter `a` using long-tail opposite learning
        a = np.mean(long_tail_opposite_learning(np.array([t / Max_iterations]), 0, 1, current_distribution))

        # Calculate the mutation rate for adaptive mutation using the current long-tail distribution
        mutation_rate = (np.random.pareto(2.0) + 1) * adaptive_parameter(t, Max_iterations, initial_mutation_rate, final_mutation_rate)

        for i in range(SearchAgents):
            # Wolf movement
            r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
            A1, A2, A3 = 2 * a * r1 - a, 2 * a * r2 - a, 2 * a * r3 - a
            C1, C2, C3 = 2 * r1, 2 * r2, 2 * r3

            # Wolf movement towards Alpha
            D_alpha = abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha

            # Wolf movement towards Beta
            D_beta = abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta

            # Wolf movement towards Delta
            D_delta = abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta

            # Adaptive combination
            weight_alpha = adaptive_parameter(t, Max_iterations, 0.7, 0.5)
            weight_beta = adaptive_parameter(t, Max_iterations, 0.2, 0.3)
            weight_delta = adaptive_parameter(t, Max_iterations, 0.1, 0.2)
            wolves[i] = (X1 * weight_alpha + X2 * weight_beta + X3 * weight_delta)

            # Apply Adaptive Mutation
            if np.random.rand() < mutation_rate:
                random_indexed = random.randint(0, len(wolves[i]) - 1)
                cross_thrs = random.choice([1, -1])
                snum = (1 - np.random.uniform(0, 1) ** ((1 - (t / Max_iterations))) ** (1 / 2))
                
                if cross_thrs == 1:
                    wolves[i][random_indexed] -= (upperbound - wolves[i][random_indexed]) * snum
                elif cross_thrs == -1:
                    wolves[i][random_indexed] -= (wolves[i][random_indexed] - lowerbound) * snum

            # Limit the wolf's position within the search bounds if necessary
            wolves[i] = np.maximum(wolves[i], lowerbound)
            wolves[i] = np.minimum(wolves[i], upperbound)

            # Apply Long-Tail Opposite Learning Based (LTOL)
            if np.random.rand() < olb_probability:
                wolves[i] = long_tail_opposite_learning(wolves[i], lowerbound, upperbound, current_distribution)

            # Evaluate fitness
            fitness_i = fitness(wolves[i])

            # Update Alpha, Beta, and Delta if a better objective value is found
            if fitness_i < alpha_fitness:
                alpha = wolves[i].copy()
                alpha_fitness = fitness_i
                Alpha_pos_history.append(alpha.copy())  # Add Alpha position to history
                no_improvement_counter = 0  # Reset counter on improvement
            elif alpha_fitness < fitness_i < beta_fitness:
                beta = wolves[i].copy()
                beta_fitness = fitness_i
            elif beta_fitness < fitness_i < delta_fitness:
                delta = wolves[i].copy()
                delta_fitness = fitness_i
            else:
                no_improvement_counter += 1  # Increment counter if no improvement

            # Switch distribution if no improvement for a certain number of iterations
            if no_improvement_counter > 10:
                current_distribution = distribution_types[(distribution_types.index(current_distribution) + 1) % len(distribution_types)]
                no_improvement_counter = 0

        # Calculate exploration and exploitation percentages
        dimensionwise_distances = wolves - np.median(wolves, axis=0)
        dimensionwise_diversity = np.mean(dimensionwise_distances, axis=0)
        population_diversity = np.mean(dimensionwise_diversity)
        max_diversity = np.max(np.abs(dimensionwise_diversity))
        exploration_percentage = (population_diversity / max_diversity) * 100
        exploitation_percentage = (np.abs(population_diversity - max_diversity) / max_diversity) * 100
        Exploration[t] = exploration_percentage
        Exploitation[t] = exploitation_percentage

        # Store history
        trajectories.append(alpha.copy())
        fitness_history.append(alpha_fitness)
        position_history.append(wolves.copy())

    # Final results
    best_solution = alpha
    best_fitness = alpha_fitness

    return best_fitness, best_solution, fitness_history, trajectories, position_history, Exploration, Exploitation
