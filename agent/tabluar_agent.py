import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from agent.abstract_agent import AbstractAgent
from agent.fix_strategy_agent import StrategyAgent
from utils import argmax, label_encode
from env import Environment

class TabularAgent(AbstractAgent):
    # h is every agents' most recent h actions are visiable to others which is composed to state
    def __init__(self, name, config):
        super(TabularAgent, self).__init__(config)
        self.config = config
        self.name = name
        self.n_actions = config.n_actions
        self.own_memory = torch.zeros((config.n_episodes * 1000,))
        self.opponent_memory = torch.zeros((config.n_episodes * 1000,))
        self.Q_table = torch.zeros((2 ** config.h, 2))
        # self.Q_table = torch.full((2**config.h, 2), float('-inf'))
        self.state = None
        self.play_epsilon = config.play_epsilon
        self.policy = self.EpsilonPolicy(self.Q_table, self.play_epsilon, self.config.n_actions)         # an object

    def act(self, oppo_agent):
        # get opponent's last move
        self.opponent_action = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.config.h: oppo_agent.play_times])
        # label encode
        self.state = label_encode(self.opponent_action)
        if self.play_times < self.config.h:
            self.state = None
        return int(self.select_action())

    def update(self, reward, own_action, opponent_action):
        super(TabularAgent, self).update(reward)
        self.own_memory[self.play_times - 1] = own_action
        self.opponent_memory[self.play_times - 1] = opponent_action
        # epsilon decay
        if self.play_epsilon > self.config.min_epsilon:
            self.play_epsilon *= self.config.epsilon_decay

        if self.state is not None:
            self.next_state = label_encode(torch.cat([self.opponent_action[1:], torch.as_tensor([opponent_action])]))
            if self.name == 'QLearning':
                # Q learning
                self.Q_table[self.state, own_action] = self.Q_table[self.state, own_action] + self.config.alpha * \
                                                       (reward + self.config.discount * (torch.max(self.Q_table[self.next_state])) - self.Q_table[self.state, own_action])
            elif self.name == 'MCLearning':
                # Monte Carlo every-step q update, MC method need to sample, therefore, we use the best stationary strategy "TitforTat" to play against MC
                oppo_agent = StrategyAgent('TitForTat', self.config)
                self.mc_update(oppo_agent)


    def select_action(self):
        # epsilon greedy policy
        self.policy.set_epsilon(self.play_epsilon)
        a = self.policy.sample_action(self.state)
        return a

    def mc_sample(self, policy, oppo_agent):
        """
            A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
            and dones from environment's step function and policy's sample_action function as lists.
            Args:
            policy: A policy which allows us to sample actions with its sample_action method.

        Returns:
            Tuple of lists (states, actions, rewards, dones). All lists should have same length.
            Hint: Do not include the state after the termination in the list of states.
        """
        env_sample = Environment(self.config)
        states, actions, rewards = [], [], []
        state = None
        for i in range(self.config.h):
            a1 = policy.sample_action(state)
            a2 = oppo_agent.act_sample()
            _, r1, r2 = env_sample.step(a1, a2)
            oppo_agent.update(r2, a2, a1)
        state = label_encode(oppo_agent.own_memory)
        for i in range(min(self.config.n_episodes,20)):
            a1 = policy.sample_action(state)
            a2 = oppo_agent.act_sample()
            _, r1, r2 = env_sample.step(a1, a2)
            oppo_agent.update(r2, a2, a1)
            states.append(state)
            actions.append(a1)
            rewards.append(r1)
            state = label_encode(oppo_agent.own_memory[i+1:])
        return states, actions, rewards
    def mc_update(self, oppo_agent):
        G = 0
        states, actions, rewards = self.mc_sample(self.policy, oppo_agent)
        for idx, state in reversed(list(enumerate(states))):
            states.pop(idx)
            G = self.config.discount * G + rewards[idx]
            self.Q_table[state, actions[idx]] = self.Q_table[self.state, actions[idx]] + self.config.alpha * \
                                                (G - self.Q_table[self.state, actions[idx]])

    def show(self):
        print("==================================================")
        start = 0
        if self.play_times > 36:
            start = self.play_times - 36
        print(f'{self.name} play {self.play_times} rounds\nQ_table:\n{self.Q_table}\nYour action: {self.own_memory[start:self.play_times]}\nOppo action: {self.opponent_memory[start:self.play_times]}')

