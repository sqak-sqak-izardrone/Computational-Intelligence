import time,Route
import sys
from Maze import Maze
from PathSpecification import PathSpecification
import numpy as np
import Direction

# Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation,alpha,beta):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation
        self.alpha=alpha
        self.beta=beta
        self.optimal_path=Route()
        self.optimal_path_length=sys.maxint




     # Loop that starts the shortest path process
     # @param spec Spefication of the route we wish to optimize
     # @return ACO optimized route
    def find_shortest_route(self, path_specification):
        visited=[]
        pos=path_specification.start
        route=Route(pos)
        while not pos==path_specification.end:
            pheromones=self.maze.get_pheromone(pos)
            a_dir=[]
            prob=[[],[]]
            for dir in range(4):
                if pos.add_direction(Direction(dir)) not in visited:
                    a_dir.append(dir)
            if len(a_dir)==0:
                visited=[]
                pos=path_specification.start
                route=Route(pos)
                continue
            for i in a_dir:
                if i==0:
                    prob[0].append(i)
                    prob[1].append(np.power(pheromones.east,self.alpha))
                elif i==1:
                    prob[0].append(i)
                    prob[1].append(np.power(pheromones.north,self.alpha))
                elif i==2:
                    prob[0].append(i)
                    prob[1].append(np.power(pheromones.west,self.alpha))
                else:
                    prob[0].append(i)
                    prob[1].append(np.power(pheromones.south,self.alpha))
            sum=0
            for i in range(len(prob[1])):
                sum+=prob[1][i]
            prob[1]=prob[1]/sum
            dir=prob[0][np.random.choice(np.arange(0,len(prob[1])), p=prob[1])]
            visited.append(pos)
            pos=pos.add_direction(Direction(dir))
            route.add(Direction(dir)) 
        return route