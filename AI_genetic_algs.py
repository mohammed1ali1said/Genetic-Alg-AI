import random

class Indiv():
    def __init__(self, chromosome) -> None:
        self.chromosome = chromosome  
        self.fitness = self.calc_fitness()
 
    def calc_fitness(self) -> int:
        # Here you can define or pass a fitness function, for example:
        # Hamming distance or any domain-specific function it can also be adaptive not only static 
        return sum(self.chromosome)  # A dummy fitness calculation 
    
    def get_fitness(self) -> int:
        return self.fitness
    
    def get_chromosome(self):
        return self.chromosome
    

class GeneticAlg():
    def __init__(self, pop_size=500, num_genes=10, fitness_func=None, max_generations=100, mutation_rate=0.25, crossover_method=None, mutation_method=None, parent_selection_method=None, opt=None):
        self.pop_size = pop_size
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.parent_selection_method = parent_selection_method
        self.population = []
        self.opt = opt
         
    def initialize_population(self) -> None:
        for _ in range(self.pop_size):
            chromosome = [random.randint(0, 1) for _ in range(self.num_genes)]  # random binary chromosomes
            individual = Indiv(chromosome)
            individual.fitness = self.fitness_func(individual.chromosome, self.opt)
            self.population.append(individual)
         
    def select_parents(self) -> tuple:
        # Assuming parent_selection_method returns a tuple of parents
        return self.parent_selection_method(self.population)

    def crossover(self, parent1, parent2) -> Indiv:
        # Assuming crossover_method combines parent1 and parent2 into a child
        return self.crossover_method(parent1, parent2)

    def mutate(self, individual) -> Indiv:
        return self.mutation_method(individual, self.mutation_rate)
    
    def evolve(self):
        best_individual = None
        for generation in range(self.max_generations):
            new_population = []
            for _ in range(self.pop_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                mutated_child = self.mutate(child)
                new_population.append(mutated_child)

            # Replace the old population with the new one
            self.population = new_population

            # Find the best individual in the current generation
            best_in_generation = max(self.population, key=lambda indiv: indiv.get_fitness())

            # Track the best overall individual
            if best_individual is None or best_in_generation.get_fitness() > best_individual.get_fitness():
                best_individual = best_in_generation
            
        return best_individual  # Return the best individual found after all generations
