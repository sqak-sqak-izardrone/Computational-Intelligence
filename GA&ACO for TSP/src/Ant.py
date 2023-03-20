import random
from Route import Route
from Direction import Direction

#Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.rand = random
        self.route = Route(self.start)

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.
    def find_route(self):
        return self.route
    
    def add_dir(self, dir):
        if dir == (0, 1):
            self.route.add(Direction.south)
        elif dir == (0, -1):
            self.route.add(Direction.north)
        elif dir == (1, 0):
            self.route.add(Direction.east)
        elif dir == (-1, 0): 
            self.route.add(Direction.west)

    def remove_last(self):
        return self.route.remove_last()
    
    def length_route(self):
        return self.route.size()

