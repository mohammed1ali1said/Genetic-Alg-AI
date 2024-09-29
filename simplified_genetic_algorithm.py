
import random

# BinPackingProblem class
class BinPackingProblem:
    def __init__(self, item_sizes, bin_capacity, num_items):
        self.item_sizes = item_sizes
        self.bin_capacity = bin_capacity
        self.num_items = num_items

# Individual class representing a solution
class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = float('inf')  # Initialize with a high fitness value

    def evaluate_fitness(self, fitness_func, problem, opt):
        self.fitness = fitness_func(self.chromosome, problem, opt)
        return self.fitness

# Genetic algorithm class
class GeneticAlgorithm:
    def __init__(self, pop_size, num_genes, fitness_func, max_generations, mutation_rate, crossover_method, mutation_method, parent_selection_method, problem, opt):
        self.pop_size = pop_size
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.parent_selection_method = parent_selection_method
        self.problem = problem
        self.population = []
        self.opt = opt

    def initialize_population(self):
        for _ in range(self.pop_size):
            chromosome = [random.randint(0, 119) for _ in range(self.num_genes)]
            individual = Individual(chromosome)
            individual.evaluate_fitness(self.fitness_func, self.problem, self.opt)
            self.population.append(individual)

    def select_parents(self):
        return self.parent_selection_method(self.population)

    def crossover(self, parent1, parent2):
        return self.crossover_method(parent1, parent2)

    def mutate(self, individual):
        return self.mutation_method(individual, self.mutation_rate, self.problem.item_sizes, self.problem.bin_capacity)

    def evolve(self):
        self.initialize_population()

        for generation in range(self.max_generations):
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                child1.evaluate_fitness(self.fitness_func, self.problem, self.opt)
                child2.evaluate_fitness(self.fitness_func, self.problem, self.opt)
                new_population.extend([child1, child2])

            # Sort the population by fitness and select the best individuals
            self.population = sorted(new_population, key=lambda x: x.fitness)[:self.pop_size]

        # Return the best individual in the population
        best_individual = min(self.population, key=lambda x: x.fitness)
        return best_individual

# Fitness function
def fitness_func(chromosome, problem, opt):
    fitness = 0
    bin_sizes = [0] * (max(chromosome) + 1)
    for i, bin_index in enumerate(chromosome):
        bin_sizes[bin_index] += problem.item_sizes[i]
        if bin_sizes[bin_index] > problem.bin_capacity:
            fitness += 100  # Penalty for exceeding bin capacity
    fitness += len([size for size in bin_sizes if size > 0])  # Number of bins used
    return fitness - opt

# Simple crossover function (single-point crossover)
def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1.chromosome) - 1)
    child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
    child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
    return Individual(child1_chromosome), Individual(child2_chromosome)

# Simple mutation method
def mutation_method(individual, mutation_rate, item_sizes, bin_capacity):
    for i in range(len(individual.chromosome)):
        if random.random() < mutation_rate:
            individual.chromosome[i] = random.randint(0, bin_capacity // min(item_sizes))
    return individual

# Tournament selection
def tournament(population):
    tournament_size = 3
    tournament_contestants = random.sample(population, tournament_size)
    return min(tournament_contestants, key=lambda x: x.fitness), min(tournament_contestants, key=lambda x: x.fitness)

# Example usage
item_sizes = [random.randint(1, 10) for _ in range(120)]
bin_capacity = 150
num_items = len(item_sizes)
opt = 10  # Example optimal value

problem = BinPackingProblem(item_sizes, bin_capacity, num_items)

ga = GeneticAlgorithm(
    pop_size=50,
    num_genes=num_items,
    fitness_func=fitness_func,
    max_generations=30,
    mutation_rate=0.1,
    crossover_method=single_point_crossover,
    mutation_method=mutation_method,
    parent_selection_method=tournament,
    problem=problem,
    opt=opt
)

solution_best_fit = ga.evolve()
print("Best fitness:", solution_best_fit.fitness)
