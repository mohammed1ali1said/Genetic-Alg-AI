def Single(parent1, parent2):
    # find a random point before is one parent after is another
    crossover_point = random.randint(1, len(parent1.chromosome) - 1)
    child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
    child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]

    child1 = Individual(child1_chromosome)
    child2 = Individual(child2_chromosome)
    return child1, child2

def Two(parent1,parent2):
    # find two points in range parent1 out of range parent 2 
    crossover_point1 = random.randint(1, len(parent1.chromosome) - 2)
    crossover_point2 = random.randint(1, len(parent1.chromosome) - 2)
    min1 = min(crossover_point1,crossover_point2)
    max1 = max(crossover_point2,crossover_point1)

    child1_chromosome = parent1.chromosome[:min1] + parent2.chromosome[min1:max1]+ parent1.chromosome[max1:]
    child2_chromosome = parent2.chromosome[:min1] + parent1.chromosome[min1:max1]+ parent2.chromosome[max1:]



    child1 = Individual(child1_chromosome)
    child2 = Individual(child2_chromosome)
    return child1, child2


def Uniform(parent1,parent2):
    for i in range(len(parent1.chromosome)):
           chromosomes1 = []   
           chromosomes2 = []
           choice =random.randint(0, 1)
           if choice == 0:
              chromosomes1.append(parent1[i])
              chromosomes2.append(parent2[i])
           else:
               chromosomes1.append(parent2[i])
               chromosomes2.append(parent1[i])
    child1 = Individual(chromosomes1)
    child2 = Individual(chromosomes2)
    return child1, child2



# there are some more advanced crossover methods like cut and splice which allows different lengths for each chromosomes 
# or half uniform where we swap half of the different genes 





               
                