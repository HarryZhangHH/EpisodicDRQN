import torch
import torch.nn as nn
import random
import numpy as np
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN():
    '''
    Implementation of DQN with target network and replay buffer
    '''
    def __int__(self):
        self.optimize = DQN.optimize()

    def train(self, agent: object, batch: object):
        loss = self.optimize(agent.policy_net, agent.target_net, agent.optimizer, batch, agent.config.discount, agent.criterion)
        agent.loss.append(loss.item())

    @staticmethod
    def compute_q_vals(Q: object, states: Type.TensorType, actions: Type.TensorType):
        """
        This method returns Q values for given state action pairs.

        Args:
            Q: Q-net  (object)
            states: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: Shape: batch_size x 1
        Returns:
            A torch tensor filled with Q values. Shape: batch_size x 1.
        """
        return torch.gather(Q(states), 1, actions)

    @staticmethod
    def compute_targets(Q: object, rewards: Type.TensorType, next_states: Type.TensorType, discount_factor: float):
        """
        This method returns targets (values towards which Q-values should move).

        Args:
            Q: Q-net  (object)
            rewards: a tensor of rewards. Shape: Shape: batch_size x 1
            next_states: a tensor of states. Shape: batch_size x obs_dim
            discount_factor: discount
        Returns:
            A torch tensor filled with target values. Shape: batch_size x 1.
        """
        return rewards + discount_factor * torch.max(Q(next_states), 1)[0].view(-1, 1)

    @staticmethod
    def optimize(policy_net: object, target_net: object, optimizer: object, batch: object, discount: float, criterion):
        state, action, reward, next_state = batch.state, batch.action, batch.reward, batch.next_state
        # compute the q value
        q_val = DQN.compute_q_vals(policy_net, state, action)
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = DQN.compute_targets(target_net, reward, next_state, discount)
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
