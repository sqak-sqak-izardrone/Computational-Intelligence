import time
from Maze import Maze
from Ant import Ant
from PathSpecification import PathSpecification

# Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze: Maze, ants_per_gen, generations, q, evaporation, alpha, beta):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation
        self.alpha = alpha 
        self.beta = beta 

     # Loop that starts the shortest path process
     # @param spec Spefication of the route we wish to optimize
     # @return ACO optimized route
    def find_shortest_route(self, path_specification: PathSpecification):
        self.maze.initialize_pheromones()
        graph = self.maze.get_graph()
        start_coordinate = path_specification.get_start()
        end_coordinate = path_specification.get_end()

        visited = set()
        visited.put(start_coordinate)

        for i in range(self.generations):
            ants = self.initilizeAnts(self, path_specification)

            neighbors = graph.get_neighbors((start_coordinate.get_x(), start_coordinate.get_y()))
            
            probabilities_of_adj_nodes = {}

            ##calculate sum for denominator
            #adj_node = ((coordinates), (weight, pheromone))
            sum = 0
            for adj_node in neighbors: 
                sum +=  adj_node[1][0]**(self.alpha) * (1/adj_node[1][1])**(self.beta)

            for adj_node in neighbors: 
                probabilities_of_adj_nodes[adj_node[0]] = adj_node[1][0]**(self.alpha) * (1/adj_node[1][1])**(self.beta)/sum

            for 

        return None
    


    def initilizeAnts(self, path_specification):
        ants = []
        for i in range(self.ants_per_gen): 
            ants.append(Ant(self.maze, path_specification))
        return ants 
