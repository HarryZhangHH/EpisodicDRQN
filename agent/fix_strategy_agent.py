import random
from agent.abstract_agent import AbstractAgent

class FixStrategyAgent(AbstractAgent):
    """
    An agent using fixed strategy
    """

    def __init__(self, name: str, config: object):
        self.name = name
        self.n_actions = config.n_actions
        super(FixStrategyAgent, self).__init__(config)
        self.own_memory = []
        self.opponent_memory = []
        self.state = self.StateRepr(method=config.state_repr)
        self.policy = self.EpsilonPolicy(None, config.play_epsilon, self.n_actions)

    def roll(self):
        return random.randint(0,self.n_actions-1)

    def act(self, oppo_agent: object):
        if self.name == 'ALLC':
            return 0
        elif self.name == 'ALLD':
            return 1
        elif self.name == 'Random':
            return self.roll()
        elif self.name == 'Grudger':
            low_bound = oppo_agent.play_times-self.config.h
            low_bound = 0
            if oppo_agent.play_times == 0:
                return 0
            elif low_bound < 0:
                return clip(sum(oppo_agent.own_memory[:oppo_agent.play_times]))
            else:
                return clip(sum(oppo_agent.own_memory[low_bound : oppo_agent.play_times]))
        elif self.name == 'TitForTat':
            if oppo_agent.play_times == 0:
                return 0
            else:
                return int(oppo_agent.own_memory[oppo_agent.play_times-1])
        elif self.name == 'revTitForTat':
            if oppo_agent.play_times == 0:
                return 0
            else:
                return reverse(int(oppo_agent.own_memory[oppo_agent.play_times-1]))
        elif self.name == 'Pavlov':
            if oppo_agent.play_times == 0 or self.play_times == 0:
                return 0
            if oppo_agent.own_memory[oppo_agent.play_times-1] == self.own_memory[self.play_times-1]:
                return 0
            else:
                return 1

    def act_sample(self):
        if self.name == 'TitForTat':
            if len(self.opponent_memory) == 0:
                return 0
            else:
                return self.opponent_memory[-1]

    def update(self, reward: float, own_action: int, opponent_action: int):
        super(FixStrategyAgent, self).update(reward)
        self.own_memory.append(own_action)
        self.opponent_memory.append(opponent_action)
        self.own_action = own_action

    def determine_convergence(self, threshold: int, k: int):
        return True

    def reset(self):
        super(FixStrategyAgent, self).reset()
        self.own_memory = []
        self.opponent_memory = []

    def show(self):
        print("==================================================")
        print(f'{self.name}\nYour action: {self.own_memory[-20:]}\nOppo action: {self.opponent_memory[-20:]}')

def clip(x):
    if x >= 1:
        return 1
    else:
        return 0

def reverse(x):
    if x == 0:
        return 1
    elif x == 1:
        return 0
    
