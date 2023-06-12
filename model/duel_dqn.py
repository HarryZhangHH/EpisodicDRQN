import torch
import torch.nn as nn
import random
import numpy as np
from utils import *
from model import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DuelDQN(nn.Module):

    def __init__(self, input_size: int, out_size: int, hidden_size: int, batch_size: int, time_step: int, num_layers: int=1):
        super(DuelDQN, self).__init__()
        self.batch_size = batch_size
        self.time_step = time_step
        self.input_size = input_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
                                     stride=2)  # potential check - in_channels
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=hidden_size, kernel_size=3, stride=1)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.adv = nn.Linear(in_features=hidden_size, out_features=self.out_size)
        self.val = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x,  hidden_state, cell_state, batch_size: int = None, time_step: int = None):
        if batch_size is None:
            batch_size = self.batch_size
        if time_step is None:
            time_step = self.time_step

        x = x.view(batch_size * time_step, 1, self.input_size, self.input_size)

        conv_out = self.relu(self.conv1(x))
        # conv_out = self.relu(self.bn2(self.conv2(conv_out)))
        conv_out = self.relu(self.conv3(conv_out))

        conv_out = conv_out.view(batch_size, time_step, self.hidden_size)
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.lstm.flatten_parameters()
        lstm_out = self.lstm(conv_out, (hidden_state, cell_state))
        out = lstm_out[0][:, time_step-1, :]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        adv_out = self.adv(out)
        val_out = self.val(out)
        q_out = val_out.expand(batch_size,self.out_size) + (adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(batch_size,self.out_size))

        return q_out, (h_n, c_n)

    def init_hidden_states(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).float().to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).float().to(device)
        return h, c

    def train(self, agent: object, batch: object, hidden_state: Type.TensorType, cell_state: Type.TensorType):
        loss = self.optimize(agent.policy_net, agent.target_net, agent.optimizer, batch, agent.config.discount, agent.criterion, hidden_state, cell_state)
        agent.loss.append(loss.item())
        return loss

    @staticmethod
    def compute_q_vals(Q: object, states: Type.TensorType, actions: Type.TensorType, hidden_state: Type.TensorType, cell_state: Type.TensorType):
        """
        This method returns Q values for given state action pairs.

        Args:
            Q: Q-net  (object)
            states: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: Shape: batch_size x 1
        Returns:
            A torch tensor filled with Q values. Shape: batch_size x 1.
        """
        q_vals, _ = Q(states, hidden_state, cell_state)
        return torch.gather(q_vals, 1, actions)

    @staticmethod
    def compute_targets(Q: object, rewards: Type.TensorType, next_states: Type.TensorType, discount_factor: float, hidden_state: Type.TensorType, cell_state: Type.TensorType):
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
        q_next, _ = Q(next_states, hidden_state, cell_state)
        return rewards + discount_factor * torch.max(q_next, 1)[0].view(-1, 1)

    @staticmethod
    def optimize(policy_net: object, target_net: object, optimizer: object, batch: object, discount: float, criterion: object, hidden_state: Type.TensorType, cell_state: Type.TensorType):
        state, action, reward, next_state = batch.state, batch.action, batch.reward, batch.next_state
        # compute the q value
        q_val = DuelDQN.compute_q_vals(policy_net, state, action, hidden_state, cell_state)
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = DuelDQN.compute_targets(target_net, reward, next_state, discount, hidden_state, cell_state)
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
