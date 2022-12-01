import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_hidden):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, num_hidden),  # hidden layer
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, output_size)
        )

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(-1, self.input_size)
        logits = self.linear_relu_stack(x)
        return logits

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
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
        out = self.fc(out[:, -1, :])
        return out

class A2CNetwork(nn.Module):

    def __init__(self, input_size, output_size, num_hidden):
        super(A2CNetwork, self).__init__()
        # predict how much reward the agent will receive until the end of the episode
        self.input_size = input_size
        self.critic = nn.Sequential(
            nn.Linear(input_size, num_hidden),  # hidden layer
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )
        #  pick actions to perform
        self.actor = nn.Sequential(
            nn.Linear(input_size, num_hidden),  # hidden layer
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(-1, self.input_size)
        value = self.critic(x)
        action_prob = self.actor(x)
        return value, action_prob

    def get_critic(self, x):
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(-1, self.input_size)
        return self.critic(x)

    def evaluate_action(self, state, action):
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

    def act(self, state):
        value, actions = self.forward(state)
        m = torch.distributions.Categorical(actions)
        return m.sample().item()

class A2CLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(A2CLSTM, self).__init__()
        # predict how much reward the agent will receive until the end of the episode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.critic_layer = nn.Linear(hidden_size, 1)
        self.actor_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
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

    def get_critic(self, x):
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(x.size(0), -1, self.input_size)
        value, _ = self.forward(x)
        return value

    def evaluate_action(self, state, action):
        """
        Returns
        -------
        value: (float tensor) the expected value of state
        log_probs: (float tensor) the log probability of taking the action in the state
        entropy: (float tensor) the entropy of each state's action distribution
        """
        state = state.view(state.size(0), -1, self.input_size)
        values, action_prob = self.forward(state)
        m = torch.distributions.Categorical(action_prob)
        log_probs = m.log_prob(action).view(-1, 1)
        entropy = m.entropy().mean()
        return values, log_probs, entropy

    def act(self, state):
        state = state.view(state.size(0), -1, self.input_size)
        value, actions = self.forward(state)
        m = torch.distributions.Categorical(actions)
        return m.sample().item()