import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init
import numpy as np
from utils import Type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class A2CLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(A2CLSTM, self).__init__()
        # predict how much reward the agent will receive until the end of the episode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.critic_layer = nn.Linear(hidden_size, 1)
        self.actor_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: Type.TensorType):
        # Set initial hidden and cell states
        # x need to be: (batch_size, seq_length, input_size)   seq_length=config.h
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(x.size(0), -1, self.input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        value = self.critic_layer(out[:, -1, :])
        action_prob = torch.softmax(self.actor_layer(out[:, -1, :]), dim=-1)
        return value, action_prob

    def get_critic(self, x: Type.TensorType):
        value, _ = self.forward(x)
        return value

    def evaluate_action(self, state: Type.TensorType, action: Type.TensorType):
        """
        Returns
        -------
        value: (float tensor) the expected value of state
        log_probs: (float tensor) the log probability of taking the action in the state
        entropy: (float tensor) the entropy of each state's action distribution
        """
        values, action_prob = self.forward(state)
        m = torch.distributions.Categorical(action_prob)
        log_probs = m.log_prob(action).view(-1, 1)
        entropy = m.entropy().mean()
        return values, log_probs, entropy

    def act(self, state: Type.TensorType):
        value, actions = self.forward(state)
        m = torch.distributions.Categorical(actions)
        return m.sample().item()


# Randomized Ensemble Actor Critic
class FeatureNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, feature_size: int):
        super(FeatureNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(feature_size, hidden_size)
        # self.fc_bn = nn.BatchNorm1d(hidden_size * 2)
        self.reset_parameters()

    def reset_parameters(self):
        # for name, param in self.lstm.named_parameters():
        #     weight_init.uniform_(param)
        self.fc.weight.data.uniform_(*hidden_init(self.fc))

    def forward(self, x: Type.TensorStructType):
        x1, x2 = x[0], x[1]
        x1 = x1.type(torch.FloatTensor).to(device)
        x1 = x1.view(x1.size(0), -1, self.input_size)
        x2 = x2.type(torch.FloatTensor).to(device)
        x2 = x2.view(x2.size(0), -1)

        h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)
        # feature net
        out_lstm, _ = self.lstm(x1, (h0, c0))  # out_lstm: tensor of shape (batch_size, seq_length, hidden_size)
        out_fc = self.fc(x2)
        x = torch.cat((out_lstm[:, -1, :].view(x1.size(0), self.hidden_size), out_fc.view(x1.size(0), -1)), dim=1)
        x = F.relu(x)
        return x

class CriticNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        super(CriticNet, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.input_size = input_size
        # critic net
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.critic:
            layer.weight.data.uniform_(*hidden_init(layer)) if isinstance(layer, nn.Linear) else None

    def forward(self, x: Type.TensorType):
        """
        Returns
        -------
        value: (float tensor) the expected value of state
        """
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(-1, self.input_size)
        value = self.critic(x)
        return value

class ActorNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 42):
        super(ActorNet, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.input_size = input_size
        # actor net
        self.actor = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size),  # hidden layer
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            torch.nn.Softmax(dim=-1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.actor:
            layer.weight.data.uniform_(*hidden_init(layer)) if isinstance(layer, nn.Linear) else None


    def forward(self, x: Type.TensorType):
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(-1, self.input_size)
        action_prob = self.actor(x)
        return action_prob

    def evaluate_action(self, state: Type.TensorType, action: Type.TensorType):
        """
        Returns
        -------

        log_probs: (float tensor) the log probability of taking the action in the state
        entropy: (float tensor) the entropy of each state's action distribution
        """
        action_prob = self.forward(state)
        m = torch.distributions.Categorical(action_prob)
        log_probs = m.log_prob(action).view(-1, 1)
        entropy = m.entropy().mean()
        return log_probs, entropy

    def act(self, state: Type.TensorType):
        """
        Returns
        -------
        action: (int) the sampled action
        entropy: (float tensor) the entropy of each state's action distribution
        """
        action_prob = self.forward(state)
        m = torch.distributions.Categorical(action_prob)
        entropy = m.entropy().mean()
        action = m.sample().item()
        return action, entropy