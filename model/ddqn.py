import torch
import torch.nn as nn
import random
import numpy as np
from utils import *
from model import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDQN(DQN):
    '''
    Implementation of DDQN with target network and replay buffer
    '''
    def __init__(self):
        self.optimize = DDQN.optimize()

    @staticmethod
    def compute_targets(Q: object, Q_target: object , rewards: Type.TensorType, next_states: Type.TensorType, discount_factor: float):
        with torch.no_grad():
            best_actions = Q(next_states).argmax(1).unsqueeze(1)
            q_target = rewards + discount_factor * torch.gather(Q_target(next_states), 1, best_actions)
        return q_target

    @staticmethod
    def optimize(policy_net: object, target_net: object, optimizer: object, batch: object, discount: float, criterion):
        state, action, reward, next_state = batch.state, batch.action, batch.reward, batch.next_state
        # compute the q value
        q_val = DQN.compute_q_vals(policy_net, state, action)
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = DDQN.compute_targets(policy_net, target_net, reward, next_state, discount)
        loss = criterion(q_val, target)

        # backpropagation of loss to Neural Network (PyTorch magic)
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        # for param in agent.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
        optimizer.step()
        return loss
