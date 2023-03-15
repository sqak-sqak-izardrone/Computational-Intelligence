class Node: 
    def __init__(self, coordinates):
        self.coordinates = coordinates 
        self.neighbors = {}

    def get_coordinates(self):
        return self.coordinates
    
    def add_neighbor(self, neighbor, weight, pheromone):
        self.neighbors[neighbor] = (weight, pheromone)

    def get_neighbors(self):
        return list(self.neighbors.items())
    
    def __repr__(self):
        return f"Node({self.coordinates})"