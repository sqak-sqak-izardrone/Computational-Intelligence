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
    
    def update_pheromone(self, start_point, next_point, new_pheromone): 
        weight_pheromone = list(self.nodes[start_point].neighbors[next_point])
        weight_pheromone[1] = new_pheromone
        self.nodes[start_point].neighbors[next_point] = tuple(weight_pheromone)