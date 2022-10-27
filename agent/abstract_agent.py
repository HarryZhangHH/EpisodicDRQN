import random
import torch
from utils import argmax
import numpy as np
class AbstractAgent():
    """ 
    Abstract an agent (superclass)
    Constructor. Called once at the start of each match.
    This data will persist between rounds of a match but not between matches.
    """

    def __init__(self, config):
        self.config = config
        self.running_score = 0.0
        self.play_times = 0

    def act(self):
        pass
    __act = act

    def update(self, reward):
        self.running_score = reward + self.config.discount * self.running_score
        self.play_times += 1

    __update = update

    """
    Process the results of a round. This provides an opportunity to store data 
    that preserves the memory of previous rounds.

    Parameters
    ----------
    my_strategy: bool
    other_strategy: bool
    """
    def process_results(self, my_strategy, other_strategy):
        pass

    class EpsilonPolicy(object):
        """
        A simple epsilon greedy policy.
        """
        def __init__(self, Q, epsilon, n_actions):
            self.Q = Q
            self.epsilon = epsilon
            self.n_actions = n_actions
        def sample_action(self, obs):
            """
            This method takes a state as input and returns an action sampled from this policy.

            Args:
                obs: current state (float tensor)

            Returns:
                An action (int).
            """
            if obs is None:
                return random.randint(0, self.n_actions-1)
            prob = random.random()
            if prob > self.epsilon:
                a = argmax(self.Q[obs])
            else:
                a = random.randint(0, self.n_actions-1)
            return a

        def set_epsilon(self, epsilon):
            self.epsilon = epsilon