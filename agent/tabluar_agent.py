import random
import torch
from agent.abstract_agent import AbstractAgent, ReplayBuffer
from agent.fix_strategy_agent import StrategyAgent
from utils import argmax, label_encode
from env import Environment

class TabularAgent(AbstractAgent):
    # h is every agents' most recent h actions are visiable to others which is composed to state
    def __init__(self, name, config):
        """
        Parameters
        ----------
        name
        config
        state_repr

        State and EpsilonPolicy are class object
        """
        super(TabularAgent, self).__init__(config)
        self.MADTHRESHOLD = 3
        self.config = config
        self.name = name
        self.n_actions = config.n_actions
        self.own_memory = torch.zeros((config.n_episodes * 1000,))
        self.opponent_memory = torch.zeros((config.n_episodes * 1000,))
        self.State = self.StateRepr(method=config.state_repr, mad_threshold=self.MADTHRESHOLD)              # an object
        self.Q_table = torch.zeros((2 ** config.h * self.State.len(), 2))
        # self.Q_table = torch.full((2**config.h, 2), float('-inf'))
        self.play_epsilon = config.play_epsilon
        self.Policy = self.EpsilonPolicy(self.Q_table, self.play_epsilon, self.config.n_actions)         # an object
        self.memory = ReplayBuffer(10000)

    def act(self, oppo_agent):
        # get opponent's last move
        self.opponent_action = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.config.h: oppo_agent.play_times])
        # label encode
        if self.play_times >= self.config.h:
            self.State.state = self.State.state_repr(self.opponent_action)
        return int(self.select_action())

    def update(self, reward, own_action, opponent_action):
        super(TabularAgent, self).update(reward)
        self.own_memory[self.play_times - 1] = own_action
        self.opponent_memory[self.play_times - 1] = opponent_action
        self.State.oppo_memory = self.opponent_memory[:self.play_times]
        # epsilon decay
        if self.play_epsilon > self.config.min_epsilon:
            self.play_epsilon *= self.config.epsilon_decay

        if self.State.state is not None:
            self.State.next_state = self.State.state_repr(torch.cat([self.opponent_action[1:], torch.as_tensor([opponent_action])]))
            # push the transition into ReplayBuffer
            self.memory.push(self.State.state, own_action, self.State.next_state, reward)
            if self.name == 'QLearning':
                # Q learning
                self.Q_table[self.State.state, own_action] = self.Q_table[self.State.state, own_action] + self.config.alpha * \
                                                       (reward + self.config.discount * (torch.max(self.Q_table[self.State.next_state])) - self.Q_table[self.State.state, own_action])

            # elif self.name == 'MCLearning':
            #     # Monte Carlo every-step q update, MC method need to sample, therefore, we use the best stationary strategy "TitforTat" to play against MC
            #     oppo_agent = StrategyAgent('TitForTat', self.config)
            #     self.mc_update(oppo_agent)


    def select_action(self):
        # epsilon greedy policy
        self.Policy.set_epsilon(self.play_epsilon)
        a = self.Policy.sample_action(self.State.state)
        return a

    def mc_update(self):
        # Update, first-visit
        state_buffer = []
        reward_buffer = list(sub[3] for sub in self.memory.memory)
        for idx,me in enumerate(self.memory.memory):
            state, action, reward = me[0], me[1], me[3]
            if state not in state_buffer:
                G = sum(reward_buffer[idx:])
                self.Q_table[state, action] = self.Q_table[state, action] + self.config.alpha * \
                                              (G - self.Q_table[state, action])
                state_buffer.append(state)

    def reset(self):
        # reset all attribute values expect Q table for episode-end game
        super(TabularAgent, self).reset()
        self.own_memory = torch.zeros((self.config.n_episodes * 1000,))
        self.opponent_memory = torch.zeros((self.config.n_episodes * 1000,))
        self.play_epsilon = (self.config.play_epsilon + self.play_epsilon)*0.3
        self.State = self.StateRepr(method=self.config.state_repr, mad_threshold=self.MADTHRESHOLD)
        self.Policy = self.EpsilonPolicy(self.Q_table, self.play_epsilon, self.config.n_actions)  # an object
        self.memory.clean()


    def show(self):
        print("==================================================")
        start = 0
        if self.play_times > 36:
            start = self.play_times - 36
        print(f'{self.name} play {self.play_times} rounds\nQ_table:\n{self.Q_table}\nYour action: {self.own_memory[start:self.play_times]}\nOppo action: {self.opponent_memory[start:self.play_times]}')

