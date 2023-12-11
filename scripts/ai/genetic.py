import numpy as np
"""
In this algorithm, the goal is to train an agent to play the game and learning through a genetic algorithm.

The general approach is that given a game state, we will have a policy for the agent to take certain actions
(in this case it is simple as it is a binary operation of up or not up). 

We can keep track of all the actions taken per tick and the game state at that time (position of player, obstacles, score).

With that, we can then optimize the action set of the bot and it can make a calculation per tick on whether to go up or not.
"""

epochs = 10

generation = []


agent = [0, 0, 0, 1]
# position of player, obstacle, projectile


STATE_SIZE = 6

len(agent)
agent = agent, len(agent)

class Species:
    '''
    Here, only store the values specific to each player. The rest of the state goes with the game
    and is passed in as a parameter to the choose_action function.
    State array values:
    0: y position
    1: y velocity
    '''
    def __init__(self, generation_size):
        self.generation_size = generation_size
        self.players = np.random.rand(generation_size, STATE_SIZE) # the genes for each player in the generation
        self.states = np.zeros((generation_size, 2)) # the state for each player in the generation
        self.scores = np.zeros((generation_size)) # score of each player in the generation


    def choose_action(self, state: np.ndarray) -> np.ndarray:
        actions = self.players @ state
        # perform relu
        actions = np.maximum(actions, 0)
        # only activate when > 0.5
        return np.where(actions > 0.5, 1, 0)

generation_size = 10
generation.append(agent)