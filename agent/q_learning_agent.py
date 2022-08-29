import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from agent.abstract_agent import AbstractAgent

Transition = namedtuple('Transition', ['state','action','next_state','reward'])

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


class NeuralNetwork(nn.Module):

    def __init__(self, h, outputs):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2*h, 256),  # hidden layer
            nn.ReLU(),
            nn.Linear(256, outputs)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        

class QLearningAgent(AbstractAgent):
    # h is every agents' most recent h actions are visiable to others which is composed to state
    def __init__(self, config, name='QLearning'):
        super(QLearningAgent, self).__init__(config)
        self.name = name
        self.n_actions = config.n_actions
        self.own_memory = torch.zeros((config.n_episodes, ))
        self.opponent_memory = torch.zeros((config.n_episodes, ))
        self.Q_table = torch.full((2**config.h, 2), float('-inf'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state = None
        if config.n_episodes < 10:
            self.h = 1
        else:
            self.h = config.h
        
    # def build(self):
    #     self.policy_net = NeuralNetwork(self.h, self.n_actions).to(self.device)
        # target_net = NeuralNetwork(self.config, self.n_actions).to(self.device)
        # target_net.load_state_dict(policy_net.state_dict())
        # target_net.eval()
        # memory = ReplayMemory(10000)

    def act(self):
        n_random = (2**self.h)*self.n_actions
        # the last h actions of the opponent
        self.state = decode_one_hot(self.opponent_memory[self.play_times-self.h : self.play_times])
        if self.play_times < self.h:
            self.state = None
        if self.play_times <= n_random:
            return int(self.select_action(self.state, True))
        else:
            return int(self.select_action(self.state))

    
    def update(self, reward, own_action, opponent_action):
        super(QLearningAgent, self).update(reward)
        self.own_memory[self.play_times-1] = own_action
        self.opponent_memory[self.play_times-1] = opponent_action
        if self.state is not None:
            if self.Q_table[self.state, own_action] != float('-inf'):
                # Q learning
                self.Q_table[self.state, own_action] = (1-self.config.alpha)*self.Q_table[self.state, own_action]\
                    + self.config.alpha*(reward + self.config.discount*inf_0(torch.max(self.Q_table[opponent_action])))
            else:
                self.Q_table[self.state, own_action] = self.config.alpha*(reward + self.config.discount*inf_0(torch.max(self.Q_table[opponent_action])))

    def select_action(self, state, random_flag=False):
        sample = random.random()
        self.state = state
        if sample > self.config.play_epsilon and not random_flag:
            return torch.argmax(self.Q_table[self.state])
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
    
    def show(self):
        print(f'Q_table:\n{self.Q_table}\nYour action: {self.own_memory}\nOppo action: {self.opponent_memory}')

def decode_one_hot(state):
    decode = 0
    for i in range(state.shape[0]):
        decode += state[i]*2**i
    if type(decode) == int:
        decode = torch.tensor(decode)
    return decode.long()
        
def inf_0(x):
    # change -inf to 0
    if x == float('-inf'):
        return 0
    else:
        return x




