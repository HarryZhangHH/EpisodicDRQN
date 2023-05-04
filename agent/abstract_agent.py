import random
import torch
from utils import argmax, label_encode, Type
from collections import namedtuple, deque
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AbstractAgent():
    """ 
    Abstract an agent (superclass)
    Constructor. Called once at the start of each match.
    This data will persist between rounds of a match but not between matches.
    """

    def __init__(self, config: object):
        self.config = config
        self.running_score = 0.0
        self.play_times = 0
        self.n_actions = config.n_actions

    def act(self):
        pass
    __act = act

    def update(self, reward: float):
        self.running_score = reward + self.config.discount * self.running_score
        self.play_times += 1
    __update = update

    def optimize(self, action: int, reward: float, oppo_agent: object, state: Type.TensorType = None, flag: bool = True):
        pass
    __optimize = optimize

    def reset(self):
        self.running_score = 0.0
        self.play_times = 0
    __reset = reset

    def process_results(self, my_strategy, other_strategy):
        """
        Process the results of a round. This provides an opportunity to store data
        that preserves the memory of previous rounds.

        Parameters
        ----------
        my_strategy: bool
        other_strategy: bool
        """
        pass

    def determine_convergence(self, threshold: int, k: int):
        """
        Check the convergence of the algorithm from the empirical distribution of actions
        """
        return True

    class EpsilonPolicy(object):
        """
        A simple epsilon greedy policy.
        """
        def __init__(self, Q, epsilon: float, n_actions: int):
            """
            Parameters
            ----------
            Q: can be a tensor or a object (Neural Network)
            """
            self.Q = Q
            self.epsilon = epsilon
            self.n_actions = n_actions
        def sample_action(self, obs: Type.TensorType, q_vals: Type.TensorType = None):
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
                if q_vals is not None:
                    a = argmax(q_vals)
                elif torch.is_tensor(self.Q):
                    a = argmax(self.Q[obs])
                else:
                    self.Q.eval()
                    if not isinstance(obs, tuple):
                        obs = obs[None]
                    else:
                        obs = (obs[0][None], obs[1][None])    # used by LSTMVariant network 
                    a = argmax(self.Q(obs))
                    self.Q.train()
            else:
                a = random.randint(0, self.n_actions-1)
            return a

        def update_epsilon(self, config:object):
            if self.epsilon is None:
                return None
            # epsilon decay
            if self.epsilon > config.min_epsilon:
                self.epsilon *= config.epsilon_decay
            if self.epsilon <= config.min_epsilon:
                self.epsilon = config.min_epsilon

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
        def __init__(self, method: str = None, mad_threshold: int = 1):
            self.state = None
            self.next_state = None
            self.method = method
            self.mad_threshold = mad_threshold
            self.mad = False
            self.oppo_memory = torch.zeros((1,))

        def state_repr(self, oppo_action: Type.TensorType, own_action: Type.TensorType = None):
            """
            This method takes the opponent action and your own action as input and return an encoded state

            Args:
                oppo_action: opponent recent h actions (float tensor)
                own_action: own recent h actions (float tensor)

            Returns:
                An encoded state representation (float tensor).
            """
            if 'uni' in self.method:
                if 'label' in self.method:
                    return label_encode(oppo_action)
                return oppo_action.float()
            elif 'bi' in self.method:
                assert own_action is not None, 'Make sure you input valid own_action in bi representation'
                return torch.cat((oppo_action.float(), own_action.float())) if oppo_action.size() == own_action.size() else None
            if 'label' in self.method:
                state_emb = label_encode(oppo_action)
            if self.method == 'grudgerlabel':
                self.check_mad()
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
