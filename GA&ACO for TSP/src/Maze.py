import traceback
import sys
import numpy as np
from Graph import Graph
from Route import Route
from Direction import Direction

# Class that holds all the maze data. This means the pheromones, the open and blocked tiles in the system as
# well as the starting and end coordinates.
class Maze:

    # Constructor of a maze
    # @param walls int array of tiles accessible (1) and non-accessible (0)
    # @param width width of Maze (horizontal)
    # @param length length of Maze (vertical)
    def __init__(self, walls, width, length):
        self.walls = walls
        self.length = length
        self.width = width
        self.start = None
        self.end = None
        self.graph = Graph()
        self.initialize_pheromones()

    # Initialize pheromones to a start value.
    def initialize_pheromones(self):
        for i in range(len(self.walls)):
            for j in range(len(self.walls[0])):
                if(self.walls[i][j] == 1):
                    self.graph.add_node((i,j))
        
        for node in self.graph.nodes: 
            i, j = node
            for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]: # Check adjacent cells
                ni, nj = i+di, j+dj
                if (ni,nj) in self.graph.nodes:
                    self.graph.add_edge(node, (ni, nj), 1.0, 1.0) # Add an edge between adjacent nodes
        return

    def get_graph(self):
        return self.graph

    # Reset the maze for a new shortest path problem.
    def reset(self):
        self.initialize_pheromones()

    # Update the pheromones along a certain route according to a certain Q
    # @param r The route of the ants
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_route(self, start_point, next_point, q, length_of_route):
        adj_edges_list = self.graph.get_neighbors(start_point)
        edge_between = [n for n in adj_edges_list if n[0] == next_point]
        old_pheromone = edge_between[0][1][1] ##((x,y), (weight, pheromone))
        #print("old pheromone" + str(old_pheromone))
        updated_pheromone = old_pheromone + q/(length_of_route**2)
        #print("new pheromone" + str(updated_pheromone))
        self.graph.update_pheromone(start_point, next_point, updated_pheromone)
        return 

    def evaporate(self,evaporation,filter_size=2):
        for i in range(len(self.walls)):
            for j in range(len(self.walls[0])):
                if self.walls[i][j]==1:
                    count=0
                    pheromone_density=0
                    for x in range(-filter_size,filter_size+1):
                        for y in range(-filter_size,filter_size+1):
                            ni,nj=i+x,j+y
                            if (ni,nj) in self.graph.nodes and self.walls[i+x][j+y]==1:
                                count+=1
                                for e in self.graph.get_neighbors(tuple(i+x,j+y)):
                                    if e.pheromone>1:
                                        pheromone_density+=1
                    if count>((2*filter_size+1)**2)*3/5:
                        pheromone_density/=4*count
                        for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]: # Check adjacent cells
                            ni, nj = i+di, j+dj
                            if (ni,nj) in self.graph.nodes:
                                adj_edges_list = self.graph.get_neighbors((i,j))
                                edge_between = [n for n in adj_edges_list if n[0] == (ni,nj)]
                                old_pheromone = edge_between[0][1][1]
                                new_pheromone = old_pheromone*(1/(1+np.exp(-pheromone_density-1)))
                                self.graph.update_pheromone((i,j),(ni,nj),new_pheromone)
                    else:                                               #usual evaporation in non open areas
                        for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]: # Check adjacent cells
                            ni, nj = i+di, j+dj
                            if (ni,nj) in self.graph.nodes:
                                adj_edges_list = self.graph.get_neighbors((i,j))
                                edge_between = [n for n in adj_edges_list if n[0] == (ni,nj)]
                                old_pheromone = edge_between[0][1][1]
                                new_pheromone = old_pheromone*(1-evaporation)
                                self.graph.update_pheromone((i,j),(ni,nj),new_pheromone)
        return

     # Update pheromones for a list of routes
     # @param routes A list of routes
     # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_routes(self, route: Route, q):

        start_point = (route.get_start().get_x(), route.get_start().get_y())
        next_point = (0,0)
        for r in route.get_route():
            if r == Direction.east :
                next_point = (start_point[0] + 1, start_point[1])
            elif r == Direction.west :
                next_point = (start_point[0] - 1, start_point[1])
            elif r == Direction.south: 
                next_point = (start_point[0], start_point[1] + 1)
            elif r == Direction.north: 
                next_point = (start_point[0], start_point[1] - 1)
            self.add_pheromone_route(start_point, next_point, q, route.size())
            self.add_pheromone_route(next_point, start_point, q, route.size())
            start_point = next_point 

    # Evaporate pheromone
    # @param rho evaporation factor
    def evaporate(self, rho):
       return rho 

    # Width getter
    # @return width of the maze
    def get_width(self):
        return self.width

    # Length getter
    # @return length of the maze
    def get_length(self):
        return self.length

    # Returns a the amount of pheromones on the neighbouring positions (N/S/E/W).
    # @param position The position to check the neighbours of.
    # @return the pheromones of the neighbouring positions.
    def get_surrounding_pheromone(self, position):
        return None

    # Pheromone getter for a specific position. If the position is not in bounds returns 0
    # @param pos Position coordinate
    # @return pheromone at point
    def get_pheromone(self, pos):
        return 0

    # Check whether a coordinate lies in the current maze.
    # @param position The position to be checked
    # @return Whether the position is in the current maze
    def in_bounds(self, position):
        return position.x_between(0, self.width) and position.y_between(0, self.length)

    # Representation of Maze as defined by the input file format.
    # @return String representation
    def __str__(self):
        string = ""
        string += str(self.width)
        string += " "
        string += str(self.length)
        string += " \n"
        for y in range(self.length):
            for x in range(self.width):
                string += str(self.walls[x][y])
                string += " "
            string += "\n"
        return string

    # Method that builds a mze from a file
    # @param filePath Path to the file
    # @return A maze object with pheromones initialized to 0's inaccessible and 1's accessible.
    @staticmethod
    def create_maze(file_path):
        try:
            f = open(file_path, "r")
            lines = f.read().splitlines()
            dimensions = lines[0].split(" ")
            width = int(dimensions[0])
            length = int(dimensions[1])
            
            #make the maze_layout
            maze_layout = []
            for x in range(width):
                maze_layout.append([])
            
            for y in range(length):
                line = lines[y+1].split(" ")
                for x in range(width):
                    if line[x] != "":
                        state = int(line[x])
                        maze_layout[x].append(state)
            print("Ready reading maze file " + file_path)
            return Maze(maze_layout, width, length)
        except FileNotFoundError:
            print("Error reading maze file " + file_path)
            traceback.print_exc()
            sys.exit()