import random
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import namedtuple, deque

from model import LSTMVariant
from selection.memory import UpdateMemory, ReplayBuffer
from utils import *

TARGET_UPDATE = 10
HIDDEN_SIZE = 256
BATCH_SIZE = 128
FEATURE_SIZE = 4
NUM_LAYER = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def a2c_selection(config: object, agents: dict, env: object):
    """
    A2C selection method using LSTM-Variant in sequential way

    Parameters
    ----------
    config: object
    agents: dict[object]
        dictionary of n unupdated agents
    env: object
    rnn: boolean
        default False: not use LSTM as the function approximator nextwork

    Returns
    -------
    agents: dict[object]
        dictionary of n unupdated agents
    """
    n_agents = len(agents)
    max_reward = config.temptation / (1 - config.discount)  # sum of geometric progression

    for n in agents:
        agent = agents[n]
        agent.SelectionPolicyNN = LSTMVariant(n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE * n_agents, n_agents - 1,
                                              HIDDEN_SIZE).to(device)
        agent.SelectionTargetNN = LSTMVariant(n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE * n_agents, n_agents - 1,
                                              HIDDEN_SIZE).to(device)
        agent.SelectionTargetNN.load_state_dict(agent.SelectionPolicyNN.state_dict())
        agent.SelectMemory = ReplayBuffer(10000)
        agent.SelectOptimizer = torch.optim.Adam(agent.SelectionPolicyNN.parameters(), lr=config.learning_rate)