from Node import Node
class Graph: 
    def __init__(self):
        self.nodes = {}

    def add_node(self, coordinates):
        if coordinates not in self.nodes: 
            self.nodes[coordinates] = Node(coordinates)

    def add_edge(self, src, dest, weight, pheromone):
        self.add_node(src)
        self.add_node(dest)

        self.nodes[src].add_neighbor(dest, weight, pheromone)

    def get_neighbors(self, coordinates):
        return self.nodes[coordinates].get_neighbors()