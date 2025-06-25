import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- GA Hyperparameters ---
# These hyperparameters can be adjusted to optimize the performance of the genetic algorithm.
POPULATION_SIZE = 100
GENERATIONS = 50
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
    sum1 = -a*np.exp(-b * np.sqrt(np.sum(solution**2) / d))
    sum2 = -np.exp(np.sqrt(np.sum(np.cos(2 * np.pi * solution)) / d))
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
