import random
import numpy as np
from TSPData import TSPData

class Individual:
    """
    Individual has genome (set of chromosome) and a self-caculated fitness score
    """
    def __init__(self, tsp_data, genome):
        self.genome = genome
        self.tsp_data = tsp_data
        self.fitness_score = self.fitness(tsp_data, genome)

    def fitness(self, tsp_data, genome):
        sum = tsp_data.get_start_distances()[genome[0]]
        sum += tsp_data.get_end_distances()[genome[len(genome)-1]]
        for i in range(1, len(genome)):
            sum += tsp_data.get_distances()[genome[i-1]][genome[i]]
        return sum
    def update_fitness(self):
        self.fitness_score = self.fitness(self.tsp_data, self.genome)

# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:

    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size):
        self.generations = generations
        self.pop_size = pop_size
     
    
    # def cal_fitness(self, genome):
    #     sum = 0
    #     for i in range(len(genome)):
    #         sum+=TSPData.


    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data, p_crossover, p_mutate):
        genome = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        # print("Shuffle ", random.shuffle(genome))
        population = []
        for i in range(self.pop_size):
            population.append(Individual(tsp_data, random.sample(genome, len(genome))))
        for i in range(self.generations):
            population = self.reproduce(tsp_data, p_crossover, p_mutate, population)
        population.sort(key = lambda x : x.fitness_score)
        print("final ", population[0].fitness_score)
        # the king
        return population[0].genome

    def reproduce(self, tsp_data, p_crossover, p_mutation, population):
        children = []
        population.sort(key = lambda x : x.fitness_score)

        # Elitism where we reserve two best individual for the crown
        children.append(population[0])
        children.append(population[1])
        population = population[2:]
        print("current generation best ", children[0].fitness_score)
        for i in range(1, len(population), 2):
            pairs = random.choices(population=population, weights = reversed([parents.fitness_score for parents in population]), k = 2)
            queen = pairs[0]
            king = pairs[1]
            crown_prince, prince = self.crossover(tsp_data, queen, king, p_crossover)
            children.append(crown_prince)
            children.append(prince)
        
        for child in children:
            mutation = random.uniform(0, 1)
            for i in range(len(child.genome)):
                if mutation < p_mutation:
                    mutator = i
                    while mutator == i :
                        mutator = random.randrange(18)
                    temp = child.genome[i]
                    child.genome[i] = child.genome[mutator]
                    child.genome[mutator] = temp
                    child.update_fitness()
        return children

    def crossover(self, tsp_data, queen, king, p_crossover):
        if (random.uniform(0, 1) < p_crossover):
            pivot = random.randint(1, len(queen.genome))
            crown_prince = queen.genome[:pivot]
            prince = king.genome[:pivot]

            for i in king.genome:
                if i not in crown_prince:
                    crown_prince.append(i)
            
            for i in queen.genome:
                if i not in prince:
                    prince.append(i)
            return Individual(tsp_data, crown_prince), Individual(tsp_data, prince)
        return queen, king