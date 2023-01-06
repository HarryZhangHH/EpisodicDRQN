import torch
import torch.nn as nn
from utils import Type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class A2CNetwork(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super(A2CNetwork, self).__init__()
        # predict how much reward the agent will receive until the end of the episode
        self.input_size = input_size
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # hidden layer
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        #  pick actions to perform
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # hidden layer
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: Type.TensorType):
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(-1, self.input_size)
        value = self.critic(x)
        action_prob = self.actor(x)
        return value, action_prob

    def get_critic(self, x: Type.TensorType):
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(-1, self.input_size)
        return self.critic(x)

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