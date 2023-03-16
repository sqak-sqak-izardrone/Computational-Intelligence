import time
import random
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
        for i in range(self.generations):
            ants = self.initilizeAnts(path_specification)
            #iteration  = 1
            #while iteration > 0: 
            while (len(ants) != 0):
                ##set-up before traverse the graph
                start_coordinate = (path_specification.get_start().get_x(), path_specification.get_start().get_y())
                end_coordinate = (path_specification.get_end().get_x(), path_specification.get_end().get_y())
                visited = {}
                visited[start_coordinate] = True
                for node in graph.nodes: 
                    if node != start_coordinate:
                        visited[node] = False
                ant = ants.pop()

                ## condition for stop: you can not walk anymore
                while not visited[end_coordinate]:
                    neighbors = graph.get_neighbors(start_coordinate)
                    available_neighbors = [n for n in neighbors if not visited[n[0]]]
                    
                    if not available_neighbors: 
                        #print(("Start:",start_coordinate))
                        while True:
                            prev_step = ant.remove_last()
                            #print(("step", prev_step))
                            prev_node = self.move_back(start_coordinate, prev_step)
                            #print(("prev_node", prev_node))
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
                    
                    if len(set(probabilities_of_adj_nodes.values())) == 1: 
                        ## if all the values are the same we randomly choose a path. 
                        keys_list = list(probabilities_of_adj_nodes.keys())
                        next_node = keys_list[random.randint(0, len(keys_list)-1)]
                    else: 
                        ## this max function gonna return the key of the map.
                        next_node = max(probabilities_of_adj_nodes) 
                    visited[next_node] = True                   
                    res = (next_node[0] - start_coordinate[0], next_node[1] - start_coordinate[1])
                    ant.add_route(res)
                    start_coordinate = next_node
                    
                ##updating pheromone
                self.maze.add_pheromone_routes(ant.find_route(), self.q, self.evaporation)
                #iteration -= 1
            print("One generation is finished")   
        for node in graph.nodes:
            print(node, graph.nodes[node].neighbors)
        return graph
    
    def initilizeAnts(self, path_specification):
        ants = []
        for i in range(self.ants_per_gen): 
            ants.append(Ant(self.maze, path_specification))
        return ants 
    def move_back(self, start_coordinate, prev_step): 
        if prev_step == 0: 
            return (start_coordinate[0] - 1, start_coordinate[1])
        elif prev_step == 1: 
            return (start_coordinate[0], start_coordinate[1] + 1)
        elif prev_step == 2:
            return (start_coordinate[0] + 1, start_coordinate[1])
        elif prev_step == 3:
            return (start_coordinate[0], start_coordinate[1] - 1)