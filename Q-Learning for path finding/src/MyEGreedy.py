from Agent import Agent
from Maze import Maze 
from QLearning import QLearning
import random
class MyEGreedy:
    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent: Agent, maze: Maze,end_state):
        # TODO to select an action at random in State s
        valid_actions = maze.get_valid_actions(agent,end_state)
        if len(valid_actions)==0: 
            return maze.actions[4]     
        return random.choice(valid_actions)
    
    def get_best_action(self, agent: Agent, maze: Maze, q_learning: QLearning,end_state):
        # TODO to select the best possible action currently known in State s.
        valid_actions = maze.get_valid_actions(agent,end_state)
        if len(valid_actions)==0: 
            return maze.actions[4]  
        actions_value = q_learning.get_action_values(agent.get_state(maze), valid_actions)
        max_index = actions_value.index(max(actions_value))
        return valid_actions[max_index]

    def get_egreedy_action(self, agent: Agent, maze: Maze, q_learning: QLearning, end_state, epsilon):
        # TODO to select between random or best action selection based on epsilon.
        random_number = random.randint(0, 1)
        if random_number >= 0 and random_number <= epsilon:
            return self.get_random_action(agent, maze,end_state)
        else:
            return self.get_best_action(agent, maze, q_learning,end_state)
