import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from agent.abstract_agent import AbstractAgent
from utils import argmax, label_encode

Transition = namedtuple('Transition', ['state','action','next_state','reward'])

class NeuralNetwork(nn.Module):

    def __init__(self, h, outputs, num_hidden=128):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2*h, num_hidden),  # hidden layer
            nn.ReLU(),
            nn.Linear(num_hidden, outputs)
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class DQNAgent(AbstractAgent):
    # h is every agents' most recent h actions are visiable to others which is composed to state
    def __init__(self, config, name='QLearning'):
        super(DQNAgent, self).__init__(config)
        self.name = name
        self.n_actions = config.n_actions
        self.own_memory = torch.zeros((config.n_episodes*1000, ))
        self.opponent_memory = torch.zeros((config.n_episodes*1000, ))
        self.Q_table = torch.zeros((2**config.h, 2))
        # self.Q_table = torch.full((2**config.h, 2), float('-inf'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state = None
        # for test
        self.own_action = []
        self.play_epsilon = config.play_epsilon
        
    # def build(self):
    #     self.policy_net = NeuralNetwork(self.h, self.n_actions).to(self.device)
        # target_net = NeuralNetwork(self.config, self.n_actions).to(self.device)
        # target_net.load_state_dict(policy_net.state_dict())
        # target_net.eval()
        # memory = ReplayMemory(10000)

    def act(self, oppo_agent):
        # n_random = (2**self.config.h)*self.n_actions
        n_random = self.config.h
        # the last h actions of the opponent
        # self.state = decode_one_hot(self.opponent_memory[self.play_times-self.h : self.play_times])
        self.opponent_action = torch.as_tensor(oppo_agent.own_memory[oppo_agent.play_times-self.config.h : oppo_agent.play_times])
        self.state = label_encode(self.opponent_action)
        if self.play_times < self.config.h:
            self.state = None
        if self.play_times < n_random:
            return int(self.select_action(True))
        else:
            return int(self.select_action())

    
    def update(self, reward, own_action, opponent_action):
        super(QLearningAgent, self).update(reward)
        # test
        self.own_action.append(int(own_action))
        self.own_memory[self.play_times-1] = own_action
        self.opponent_memory[self.play_times-1] = opponent_action
        if self.play_epsilon > self.config.min_epsilon:
            self.play_epsilon *= self.config.epsilon_decay
        if self.state is not None:
            self.next_state = label_encode(torch.cat([self.opponent_action[1:], torch.as_tensor([opponent_action])]))
            # Q learning
            self.Q_table[self.state, own_action] = self.Q_table[self.state, own_action] + self.config.alpha* \
                                                   (reward + self.config.discount*(torch.max(self.Q_table[self.next_state])) - self.Q_table[self.state, own_action])

    def select_action(self, random_flag=False):
        sample = random.random()
        # epsilon greedy policy
        if sample > self.play_epsilon and not random_flag:
            return argmax(self.Q_table[self.state])
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def show(self):
        print(f'Q_table:\n{self.Q_table}\nYour action: {self.own_memory[:self.play_times]}\nOppo action: {self.opponent_memory[:self.play_times]}')






