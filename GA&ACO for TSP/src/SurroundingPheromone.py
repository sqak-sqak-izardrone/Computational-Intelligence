from Direction import Direction

# Class containing the pheromone information around a certain point in the maze
class SurroundingPheromone:

    # Creates a surrounding pheromone object.
    # @param north the amount of pheromone in the north.
    # @param east the amount of pheromone in the east.
    # @param south the amount of pheromone in the south.
    # @param west the amount of pheromone in the west.
    def __init__(self, north, east, south, west):
        self.north = north
        self.south = south
        self.west = west
        self.east = east
        self.total_surrounding_pheromone = east + north + south + west

    # Get the total amount of surrouning pheromone.
    # @return total surrounding pheromone
    def get_total_surrounding_pheromone(self):
        return self.total_surrounding_pheromone

    # Get a specific pheromone level
    # @param dir Direction of pheromone
    # @return Pheromone of dir
    def get(self, dir):
        if dir == Direction.north:
            return self.north
        elif dir == Direction.east:
            return self.east
        elif dir == Direction.west:
            return self.west
        elif dir == Direction.south:
            return self.south
        else:
            return None

    def set(self, dir, val):
        if dir == Direction.north:
            self.north = val
        elif dir == Direction.east:
            self.east = val
        elif dir == Direction.west:
            self.west = val
        elif dir == Direction.south:
            self.south = val
        self.total_surrounding_pheromone = self.north + self.east + self.west + self.south   

    def evaporate(self, rho):
        self.north = rho * self.north
        self.south = rho * self.south
        self.east = rho * self.east
        self.west = rho * self.west
        self.total_surrounding_pheromone = self.north + self.east + self.west + self.south   
