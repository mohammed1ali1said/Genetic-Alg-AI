import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time


class statistics_manager:
    def __init__(self,fitnesses):
        self.fitnesses = fitnesses


    # Define the average fitness function
    def avg_fittness_generation(self):
        sum = 0
        for fitness in self.fitnesses:
            sum += fitness

        return sum/len(self.fitnesses)

    # Define the average deviation dunction for the population at each generation
    def Standard_deviation(self):
        mean = self.avg_fittness_generation()
        sum = 0
        for fitness in self.fitnesses:
            sum+= (fitness-mean)**2

        ST=np.sqrt(sum/len(self.fitnesses))


        return ST






    def norm_and_plot(self):
        norm_min = 0
        norm_max = 100
        min_val = 0
        max_val = 13

        div_factor = max_val-min_val
        normalized_fitnesses = []
        for fitness in self.fitnesses:
            normalized = ((fitness-min_val)/(div_factor))*100
            normalized_fitnesses.append(normalized)


        fig = plt.figure()
        timer = fig.canvas.new_timer(interval = 1000) #creating a timer object and setting an interval of 500 milliseconds

        timer.add_callback(close_event)
        plt.hist(normalized_fitnesses, color='blue', bins=10)
        timer.start()
        plt.show()
        return normalized_fitnesses

def close_event():
    plt.close()




# Define the fitness function
def fitness(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score


# Define the mutate function
def mutate(individual):
    target = list("Hello, world!")
    random_position = random.randint(0, len(target) - 1)  # get a random position for character to mutate
    random_char = chr(random.randint(32, 126))
    individual[random_position] = random_char
    return individual


def roulette_wheel_selection(fitnesses):

    total=sum(fitnesses)
    probs = []
    local_sum=0
    for fit in fitnesses:
      probs.append((fit/total)+local_sum)
      local_sum+=fit/total
    random.random()


    return



# Define plotting function
def plot_distribution(data,xlabel,ylabel,title):
    # Create a histogram
    sns.set(style="whitegrid")  # Set the style of the plot
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker= 'none', linestyle='-', color='b')

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()



def evaluation(genes):
    word = "Hello, world!"
    dict = {}
    geneCount ={}
    for i in range(len(word)):
        if word[i] in dict:
            dict[word[i]] += 1
        else:
            dict[word[i]] = 1

    exact = 0
    for i in range(len(word)):
        if word[i] == genes[i]:
            exact+=1

    for i in range(len(genes)):
        if genes[i] in geneCount:
            geneCount[genes[i]]+=1
        else:
            geneCount[genes[i]] = 1
    sum = 0
    for i in geneCount:
        if i in dict:
            sum += min(geneCount[i],dict[i])
    sum -=exact
    return sum






# Define the genetic algorithm
def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, mutation_rate,crossover_method):
    # Initialize the population with random individuals
    population = []
    for i in range(pop_size):
        individual = [chr(random.randint(32, 126)) for j in range(num_genes)]
        population.append(individual)

    # Evolve the population for a fixed number of generations
    generation_avg_fitnesses = [] # list that contains the average fitness of each generation
    generation_avg_SD = []
    for generation in range(max_generations):
        # Evaluate the fitness of each individual
        fitnesses = [fitness_func(individual) for individual in population]

        Statistics_Manager = statistics_manager(fitnesses)
        generation_avg_fitnesses.append(Statistics_Manager.avg_fittness_generation())
        generation_avg_SD.append(Statistics_Manager.Standard_deviation())
        print(generation)

        if generation%20==0:
            Statistics_Manager.norm_and_plot()



        # Select the best individuals for reproduction
        elite_size = int(pop_size * 0.1)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]
        # print("Gen " , generation , "Elite: ",''.join(population[len(elite_indices)]))
        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            # CROSSOVER
            if crossover_method  == "uniform":
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(num_genes)]
            elif crossover_method == "single":
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                random_index = random.randint(0, num_genes-1)  # get a random position
                child = parent1[:random_index] + parent2[random_index:]
            elif crossover_method == "two":
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                random_index1 = 0
                random_index2 = 0
                while random_index1 >= random_index2: # keep looping until getting random_index2 > random_index1
                    random_index1 = random.randint(0, num_genes - 1)  # get a random position
                    random_index2 = random.randint(0, num_genes - 1)  # get a random position

                child = parent1[:random_index1] + parent2[random_index1:random_index2] + parent1[random_index2:]
            # MUTATION
            ga_mutation = mutation_rate * sys.maxsize
            if random.random() < ga_mutation:
                mutate(child)

            offspring.append(child)
        population = elites + offspring

    plot_distribution(generation_avg_fitnesses,'Generation','AVG','Fittness AVG distribution')
    plot_distribution(generation_avg_SD,'Generation','AVG','Standart Deviation')

    # Find the individual with the highest fitness
    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    return best_individual, best_fitness


# Run the genetic algorithm and print the result
best_individual, best_fitness = genetic_algorithm(pop_size=100, num_genes=13, fitness_func=fitness, max_generations=100,mutation_rate=0.25,crossover_method="uniform")

print("Best individual:", ''.join(best_individual))
print("Best fitness:", best_fitness)

