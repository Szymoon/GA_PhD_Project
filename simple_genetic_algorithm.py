import numpy as np
import matplotlib.pyplot as plt

# --- GA Hyperparameters ---
# These hyperparameters can be adjusted to optimize the performance of the genetic algorithm.
POPULATION_SIZE = 100
GENERATIONS = 20
MUTATION_RATE = 0.01
MUTATTION_STRENGHT = 0.1
TOURNAMENT_SIZE = 5

# --- Benchamark Functions ---
def ackley(solution):
    '''Ackley function for optimization problems.
    Args:
        solution (np.ndarray): A numpy array representing the solution vector.
    Returns:
        float: The value of the Ackley function at the given solution.'''
    a = 20
    b = 0.2
    d = len(solution) # Number of dimensions
    sum1 = -a * np.exp(-b * np.sqrt(np.sum(solution**2) / d))
    sum2 = -np.exp(np.sum(np.cos(2 * np.pi * solution)) / d)
    return sum1 + sum2 + a + np.exp(1)

# --- Function Bounds ---
# For visulization purposes, we define 2 dimensions of the Ackley function.
# The bounds are defines as a list of [min_value, max_value] for each dimension.
# We will use a dictionary to store the bounds if we want to add more benchmark functions in the future.
bounds = {
    'ackley': [-32,32]
}
n_dimensions = 2 # Number of dimensions for the Ackley function

# --- GA Core Functions ---

# 1. Initialization Function
def initialize_population(pop_size, dimensions, bounds):
    '''Initialize a population of solutions.
    Args:
        pop_size (int): The size of the population.
        dimensions (int): The number of dimensions for each solution.
        bounds (list): The bounds for each dimension.
    Returns:
        np.ndarray: A numpy array representing the initial population (pop_size, dimensions).'''
    min_bound, max_bound = bounds
    population = np.random.uniform(min_bound, max_bound, (pop_size, dimensions))
    return population

# 2. Fitness Calculation Function
def calculate_fitness(population, benchmark_function):
    '''Calculate the fitness of each solution in the population.
    Args:
        population (np.ndarray): A numpy array representing the population (pop_size, dimensions).
        benchmark_function (function): The benchmark function to evaluate the fitness (e.g., ackley).
    Returns:
        np.ndarray: A 1 D numpy array representing the fitness values for each solution.'''
    fitness_scores = np.array([benchmark_function(individual) for individual in population]) # list comprehension to calculate fitness for each individual
    return fitness_scores

# 3. Selection Function
# For simplicity we will use and define only tournament selection.
def tournament_selection(population, fitness_scores, tournament_size):
    '''Select individuals from the population using tournament selection.
    Args:
        population (np.ndarray): A numpy array representing the population (pop_size, dimensions).
        fitness_scores (np.ndarray): A 1D numpy array representing the fitness scores of the population.
        tournament_size (int): The size of the tournament.
    Returns:
        np.ndarray: A numpy array representing the selected individuals.'''
    # Randomly select individuals for the tournament
    tournament_indexes = np.random.randint(0, len(population), tournament_size)
    
    # Get the individals and their fitness scores
    tournament_contestants = population[tournament_indexes]
    tournament_fitness = fitness_scores[tournament_indexes]
    
    # Find the index of the best individual in the tournament
    best_index = np.argmin(tournament_fitness)
    
    # return the winner of the tournament
    return tournament_contestants[best_index]

# 4. Crossover Function
# We will use arithmetic crossover for simplicity.
def crossover(parent1, parent2):
    '''Perform arithmetic crossover between two parents.
    Args:
        parent1 (np.ndarray): The first parent solution.
        parent2 (np.ndarray): The second parent solution.
    Returns:
        np.ndarray: A numpy array representing the offspring solution.'''
    alpha = np.random.rand()  # Random weights for crossover
    offspring = alpha * parent1 + (1 - alpha) * parent2
    return offspring

# 5. Mutation Function
def mutate(individual, mutation_rate, mutation_strength, bounds):
    '''Mutate an individual solution.
    Args:
        individual (np.ndarray): The individual solution to mutate.
        mutation_rate (float): The probability of mutation for each gene.
        mutation_strength (float): The strength of the mutation.
        bounds (list): The bounds for each dimension.
    Returns:
        np.ndarray: The mutated individual solution.'''
    
    mutated_individual = individual.copy()
    min_bound, max_bound = bounds
    
    for i in range(len(mutated_individual)):
        if np.random.rand() < mutation_rate:
            # Add random noise from a normal distribution
            noise = np.random.normal(0, mutation_strength)
            mutated_individual[i] += noise
    
    # Clip the values so they stay within the bounds
    np.clip(mutated_individual, min_bound, max_bound, out=mutated_individual)
    
    return mutated_individual

# --- Main Genetic Algorithm Function ---

if __name__ == "__main__":
    # 1. --- SETUP ---
    benchmark_function = ackley # Benchmark function to optimize
    bounds = bounds[benchmark_function.__name__] # ensure proper benchmark function name
    
    best_fitness_history = [] # Store the best fitness value for each generation
    
    # 2. --- GENERATE DATA FOR PLOTTING ---
    x_range = np.linspace(bounds[0], bounds[1], 200)
    y_range = np.linspace(bounds[0], bounds[1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Z value (fitness) for each (x, y) pair 
    Z = np.array([[benchmark_function(np.array([x, y])) for x in x_range] for y in y_range])
        
    # 3. --- INITIALIZE POPULATION ---
    current_population  = initialize_population(POPULATION_SIZE, n_dimensions, bounds)
    
    print(f"---- Starting Genetic Algorithm Optimization for {benchmark_function.__name__} ---")
    
    # 4. --- PLOT INITIAL POPULATION ---
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Fitness Value')
    
    pop_x = current_population[:, 0]
    pop_y = current_population[:, 1]
    plt.scatter(pop_x, pop_y, color='red', label='First Generation')
    
    plt.title(f'Initial Population for {benchmark_function.__name__}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show(block=False)
    
    # 5. --- MAIN LOOP ---
    for generation in range(GENERATIONS):
        # Calculate fitness for current population
        fitness_scores = calculate_fitness(current_population, benchmark_function)
        
        # Store the best fitness value for this generation
        best_fitness = np.min(fitness_scores)
        best_fitness_history.append(best_fitness)
        
        # Print progress
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"Generation {generation + 1}/{GENERATIONS}, Best Fitness: {best_fitness:.4f}")
        
        # Create a new population
        new_population = []
        for _ in range(POPULATION_SIZE):
            # Select parents
            parent1 = tournament_selection(current_population, fitness_scores, TOURNAMENT_SIZE)
            parent2 = tournament_selection(current_population, fitness_scores, TOURNAMENT_SIZE)
            
            # Crossover to create offspring
            offspring = crossover(parent1, parent2)
            
            # Mutate the offspring
            mutated_offspring = mutate(offspring, MUTATION_RATE, MUTATTION_STRENGHT , bounds)
            
            # Add the mutated offspring to the new population
            new_population.append(mutated_offspring)
        
        # Update the current population with the new population
        current_population = np.array(new_population)
        
    # 6. --- RESULTS AND VISUALIZATION ---
    print(f"---- Genetic Algorithm Optimization Completed ----")
    
    # Final fitness scores
    final_fitness_scores = calculate_fitness(current_population, benchmark_function)
    best_solution_index = np.argmin(final_fitness_scores)
    best_solution = current_population[best_solution_index]
    best_fitness = final_fitness_scores[best_solution_index]
    
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness:.4f}")
    
    # Plotting the convergence curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(GENERATIONS), best_fitness_history, marker='o', linestyle='-', color='r')
    plt.title('Convergence Curve')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.grid()
    plt.show()
    