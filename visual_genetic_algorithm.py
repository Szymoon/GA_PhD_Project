import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# --- GA Hyperparameters ---
# These hyperparameters can be adjusted to optimize the performance of the genetic algorithm.
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.01
MUTATTION_STRENGHT = 0.1
TOURNAMENT_SIZE = 5

PLOT_DELAY = 50  # Delay in milliseconds for animation

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

# --- Main Genetic Algorithm Function and Visualization ---

if __name__ == "__main__":
    # 1. --- SETUP GA AND PLOT ---
    
    # Initialize GA state
    benchmark_function = ackley                                                         # Benchmark function to optimize
    bounds = bounds[benchmark_function.__name__]                                        # Ensure proper benchmark function name
    current_population  = initialize_population(POPULATION_SIZE, n_dimensions, bounds)  # Initial population
    best_fitness_history = []                                                           # Store the best fitness value for each generation
    
    # Create figure and subplot
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection='3d')  # 3D plot for the fitness landscape
    ax2 = fig.add_subplot(122)  # 2D plot for the convergence
    
    # Prepare the 3D surface plot 
    x_range = np.linspace(bounds[0], bounds[1], 100)
    y_range = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    #Vectorized Z value (fitness) for each (x, y) pair
    Z = np.vectorize(lambda x, y: benchmark_function(np.array([x, y])))(X, Y)
    
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, rcount=100, ccount=100)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_zlabel('Fitness Value')
    ax1.set_title(f'Fitness Landscape for {benchmark_function.__name__}')
    
    # Placeholder for the animation
    scatter_3d = ax1.scatter([], [], [], color='red', label='Population', s=25, depthshade=True)
    line_2d = ax2.plot([],[], 'r-o')
    ax2.set_title('Convergence Curve')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Fitness Value')
    ax2.grid()
   
    # 2. --- ANIMATION ANG GA MAIN FUNCTION ---
    # Function that will be called for each frame of the animation
    def update(generation):
        global current_population # Use the global variable to access the current population
        
        # --- Run one GA generation ---
        fitness_scores = calculate_fitness(current_population, benchmark_function)
        best_fitness = np.min(fitness_scores)
        best_fitness_history.append(best_fitness)
        
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
        
        # --- Update plot for this frame ---
        # Update the 3D plot with the current population
        pop_x = current_population[:, 0]
        pop_y = current_population[:, 1]
        pop_z = calculate_fitness(current_population, benchmark_function)
        scatter_3d._offsets3d = (pop_x, pop_y, pop_z)
        
        # Update the 2D convergence plot
        line_2d[0].set_data(range(len(best_fitness_history)), best_fitness_history)
        ax2.set_xlim(0, GENERATIONS)
        # Adjust ylim dynamically
        if best_fitness_history:
            ax2.set_ylim(min(best_fitness_history) * 0.9, max(best_fitness_history) + 1)
        
        # Update the title with the current generation and best fitness
        fig.suptitle(f'GA Optimization - Generation {generation + 1}/{GENERATIONS}, Best Fitness: {best_fitness:.4f}')
        
        return scatter_3d, line_2d
    
    # --- RUN THE ANIMATION ---
    ani = animation.FuncAnimation(
        fig=fig, 
        func=update,
        frames=GENERATIONS,
        interval=PLOT_DELAY,
        blit=False, # Blit is set to False to update the entire figure, more reliable
        repeat=False
    )
    
    plt.tight_layout(rect=[0,0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()  # Show the plot and start the animation
    
    # Final fitness scores
    final_fitness_scores = calculate_fitness(current_population, benchmark_function)
    best_solution_index = np.argmin(final_fitness_scores)
    best_solution = current_population[best_solution_index]
    best_fitness = final_fitness_scores[best_solution_index]
    
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness:.4f}")

    