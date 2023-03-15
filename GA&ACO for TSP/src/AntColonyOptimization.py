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
        start_coordinate = (path_specification.get_start().get_x(), path_specification.get_start().get_y())
        end_coordinate = (path_specification.get_end().get_x(), path_specification.get_end().get_y())

        for i in range(self.generations):
            ants = self.initilizeAnts(self, path_specification)
            iteration  = 10000
            while iteration > 0: 

                while (len(ants) != 0):
                    visited = set()
                    visited.put(start_coordinate)
                    ant = ants.pop()
                    ## condition for stop: you can not walk anymore
                    while end_coordinate not in visited or len(graph.get_neighbors((start_coordinate[0], start_coordinate[1])) != 0):
                        neighbors = graph.get_neighbors((start_coordinate.get_x(), start_coordinate.get_y()))
                        probabilities_of_adj_nodes = {}
                        ##calculate sum for denominator
                        #adj_node = ((coordinates), (weight, pheromone))
                        sum = 0
                        for adj_node in neighbors: 
                            if adj_node not in visited:
                                sum +=  adj_node[1][0]**(self.alpha) * (1/adj_node[1][1])**(self.beta)
                        for adj_node in neighbors: 
                            if adj_node not in visited:
                                probabilities_of_adj_nodes[adj_node[0]] = adj_node[1][0]**(self.alpha)*(1/adj_node[1][1])**(self.beta)/sum
                        
                        next_node = max(probabilities_of_adj_nodes) ## this max function gonna return the key of the map
                        visited.put(next_node)                        
                        res = (next_node[0] - start_coordinate[0], next_node[1] - start_coordinate[1])
                        ant.add_route(res)
                        start_coordinate = next_node
                        
                    ##updating pheromone
                    self.maze.add_pheromone_routes(ant.find_route(), self.q)
                
                    

            iteration -= 1

        return None
    


    def initilizeAnts(self, path_specification):
        ants = []
        for i in range(self.ants_per_gen): 
            ants.append(Ant(self.maze, path_specification))
        return ants 
