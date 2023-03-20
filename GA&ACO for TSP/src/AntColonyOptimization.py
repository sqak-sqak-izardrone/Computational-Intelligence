import time
import random
from Maze import Maze
from Ant import Ant
from PathSpecification import PathSpecification
from Route import Route
from Direction import Direction 
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
        array = []
        self.maze.initialize_pheromones()
        graph = self.maze.get_graph()
        prev_shortest_route = None 
        shortest_route = None  
        counter = 0       
        for i in range(self.generations):
            ants = self.initilizeAnts(path_specification)
            for ant in ants:
                ##set-up before traverse the graph
                start_coordinate = (path_specification.get_start().get_x(), path_specification.get_start().get_y())
                end_coordinate = (path_specification.get_end().get_x(), path_specification.get_end().get_y())
                visited = {}
                visited[start_coordinate] = True
                for node in graph.nodes: 
                    if node != start_coordinate:
                        visited[node] = False
                ## condition for stop: you can not walk anymore
                while not visited[end_coordinate]:
                    neighbors = graph.get_neighbors(start_coordinate)
                    available_neighbors = [n for n in neighbors if not visited[n[0]]]
                    
                    if not available_neighbors: 
                        while True:
                            prev_step = ant.remove_last()                           
                            prev_node = self.move_back(start_coordinate, prev_step)
                            prev_neighbors = graph.get_neighbors(prev_node)
                            available_neighbors = [n for n in prev_neighbors if not visited[n[0]]]
                            if available_neighbors:
                                start_coordinate = prev_node
                                break
                            start_coordinate = prev_node
                            
                    probabilities_of_adj_nodes = {}
                    ##calculate sum for denominator
                    #adj_node = ((coordinates), (weight, pheromone))
                    sum = 0
                                                
                    for adj_node in available_neighbors: 
                            sum +=  adj_node[1][0]**(self.alpha) * (1/adj_node[1][1])**(self.beta)
                    for adj_node in available_neighbors: 
                            probabilities_of_adj_nodes[adj_node[0]] = adj_node[1][0]**(self.alpha)*(1/adj_node[1][1])**(self.beta)/sum
                    
                    next_node = random.choices(list(probabilities_of_adj_nodes.keys()), list(probabilities_of_adj_nodes.values()))[0]
                    
                    visited[next_node] = True                   
                    res = (next_node[0] - start_coordinate[0], next_node[1] - start_coordinate[1])
                    ant.add_dir(res)
                    start_coordinate = next_node
            shortest_route = ants[0].find_route()
            ##updating pheromone
            self.maze.evaporate(self.evaporation)
            
            for ant in ants: 
                if shortest_route.size() > ant.find_route().size():
                    shortest_route = ant.find_route()
                self.maze.add_pheromone_routes(ant.find_route(), self.q)
            print(shortest_route.size())
            if prev_shortest_route is None: 
                prev_shortest_route = shortest_route
            else: 
                if(prev_shortest_route.size() > shortest_route.size()):
                    prev_shortest_route = shortest_route
                    counter += 1
            array.append(shortest_route.size())    
        return prev_shortest_route
    
    def initilizeAnts(self, path_specification):
        ants = []
        for i in range(self.ants_per_gen): 
            ants.append(Ant(self.maze, path_specification))
        return ants 
    
    def move_back(self, start_coordinate, prev_step: Direction): 
        if prev_step == Direction.east: 
            return (start_coordinate[0] - 1, start_coordinate[1])
        elif prev_step == Direction.north: 
            return (start_coordinate[0], start_coordinate[1] + 1)
        elif prev_step == Direction.west:
            return (start_coordinate[0] + 1, start_coordinate[1])
        elif prev_step == Direction.south:
            return (start_coordinate[0], start_coordinate[1] - 1)
