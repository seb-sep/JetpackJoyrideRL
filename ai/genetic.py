"""
In this algorithm, the goal is to train an agent to play the game and learning through a genetic algorithm.

The general approach is that given a game state, we will have a policy for the agent to take certain actions
(in this case it is simple as it is a binary operation of up or not up). 

We can keep track of all the actions taken per tick and the game state at that time (position of player, obstacles, score).

With that, we can then optimize the action set of the bot and it can make a calculation per tick on whether to go up or not.
"""