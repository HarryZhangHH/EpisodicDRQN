import random
import torch
from utils import argmax, label_encode
from collections import namedtuple, deque
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

    def reset(self):
        self.running_score = 0.0
        self.play_times = 0
    __reset = reset

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
            ----------
            Args:
                obs: current state (float tensor)
            ----------
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

    class StateRepr(object):
        """
        State representation, feature construction
        ----------
        Args:
            method: to select the feature construction method, choices=[None, 'grudger'] (string)
            mad_threshold: the threshold comparing the number of opponent's defection to decide when the agent will become mad (int)
        """
        def __init__(self, method=None, mad_threshold=1):
            self.state = None
            self.next_state = None
            self.method = method
            self.mad_threshold = mad_threshold
            self.mad = False
            self.oppo_memory = torch.zeros((1,))

        def state_repr(self, oppo_action, own_action=None):
            """
            This method takes the opponent action and your own action as input and return an encoded state

            Args:
                oppo_action: opponent recent h actions (float tensor)
                own_action: own recent h actions (float tensor)

            Returns:
                An encoded state representation (int).
            """
            self.check_mad()
            state_emb = label_encode(oppo_action)
            if self.method == 'grudger':
                state_emb += 2**len(oppo_action)*self.mad
            return state_emb
        def check_mad(self):
            """
                This method check the agent mad or not based on the mad_threshold
            """
            if int(torch.sum(self.oppo_memory)) > self.mad_threshold:
                self.mad = True
        def len(self):
            if self.method == 'grudger':
                return 2
            return 1

Transition = namedtuple('Transition', ['state','action','next_state','reward'])
class ReplayBuffer(object):
    """
    A replay buffer using by MC and Q-Network to store transition
    ----------
    Args:
        capacity: the capacit of replay buffer (int)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
        if len(self.memory) > self.capacity:
            self.memory = self.memory[1:]
    def clean(self):
        self.memory = deque([],maxlen=self.capacity)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)