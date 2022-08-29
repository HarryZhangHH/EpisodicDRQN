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

    def act(self):
        if self.name == 'ALLC':
            return 0
        elif self.name == 'ALLD':
            return 1
        elif self.name == 'Random':
            return self.roll()
        elif self.name == 'Grudger':
            if sum(self.opponent_memory) == 0:
                return 0
            else:
                return 1
        elif self.name == 'TitForTat':
            if len(self.opponent_memory) == 0:
                return 0
            else:
                return self.opponent_memory[-1]
    
    def update(self, reward, own_action, opponent_action):
        super(StrategyAgent, self).update(reward)
        self.own_memory.append(own_action)
        self.opponent_memory.append(opponent_action)
    

    
