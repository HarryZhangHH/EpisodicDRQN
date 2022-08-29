import random
from agent.abstract_agent import AbstractAgent

class StrategyAgent(AbstractAgent):
    """Abstract an agent
    """

    def __init__(self, name, config):
        self.name = name
        super(StrategyAgent, self).__init__(config)
        self.own_memory = []
        self.opponent_memory = []
        
    def roll(self):
        return random.randint(0,1)

    def act(self, oppo_agent):
        if self.name == 'ALLC':
            return 0
        elif self.name == 'ALLD':
            return 1
        elif self.name == 'Random':
            return self.roll()
        elif self.name == 'Grudger':
            low_bound = oppo_agent.play_times-self.config.h
            if len(oppo_agent.own_memory) == 0:
                return 0
            elif low_bound < 0:
                return clip(sum(oppo_agent.own_memory[:oppo_agent.play_times]))
            else:
                return clip(sum(oppo_agent.own_memory[low_bound : oppo_agent.play_times]))
        elif self.name == 'TitForTat':
            if len(oppo_agent.own_memory) == 0:
                return 0
            else:
                return oppo_agent.own_memory[oppo_agent.play_times-1]
    
    def update(self, reward, own_action, opponent_action, oppo_agent):
        super(StrategyAgent, self).update(reward)
        self.own_memory.append(own_action)
        self.opponent_memory.append(opponent_action)
    
    def show(self):
        print(f'Your action: {self.own_memory}\nOppo action: {self.opponent_memory}')
    
def clip(x):
    if x >= 1:
        return 1
    else:
        return 0
    
