CI Assignment 2

1. Genetic: Khoa does all. Fuck him

2. ACO
- Pseudo code -> details about implementation -> divide the tasks for implementation
- Impplement the standard algo 
- Solve the 3 advanced features 
- Tune params
- Synthesize 

3. Scheduling
- Standard MUST be done before Thursday lab 
- Advanced before Saturday
- Finish report by Monday


4. Pseudo code

Initialize the shortest path M

For each iteration:
	For each ants:
		// Encapsulate as a different function findPath(Graph)
		// P must connect the starting point with the destination
		Compute a path P according to the probabilities:
			Compute the probs to exclude visited nodes
			Walk through an edge based on the new probs, update visited nodes

		Compute length of P
		Update the delta-pheromones on each edges on P
		Compare with the shorest path M and update 

	Update pheromones using the evaporated previous pheromones and delta 

//  Do not update probs because the ants only traverse a small portion of the graph 

* Ideas for optimization
- Store probs so other ants can reuse 
- evaporation becomes a variable where less condensed pheromones evaporates more
- ants have longer visions and tends to walk straight in open areas

5. Data structure:


