from collections import namedtuple, deque
import random

class Memory(object):

    TwoAgentTransition = namedtuple('Agent',
                       ['agent_1', 'agent_2', 'action_1', 'action_2', 'reward_1', 'reward_2', 'state_1', 'state_2'])

    Record = namedtuple('Record',
                        ['agent_1', 'agent_2', 'state' , 'action_1', 'action_2', 'reward_1', 'reward_2'])

    TwoAgentFullTransition = namedtuple('Agent',
                          ['agent_1', 'agent_2', 'action_1', 'action_2', 'reward_1', 'reward_2', 'state_1', 'state_2',
                           'next_state_1', 'next_state_2'])

    Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])

    """
    Used for multi-agent games
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        pass
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    def clean(self):
        self.memory = deque([],maxlen=self.capacity)
    def __len__(self):
        return len(self.memory)

class RecordMemory(Memory):
    """
        Used for recording multi-agent games
        """
    def __init__(self, capacity: int):
        super(RecordMemory, self).__init__(capacity)

    def push(self, *args):
        self.memory.append(Memory.Record(*args))

class UpdateMemory(Memory):
    """
    Used for multi-agent games
    """
    def __init__(self, capacity: int):
        super(UpdateMemory, self).__init__(capacity)

    def push(self, *args):
        self.memory.append(Memory.TwoAgentTransition(*args))

class SettlementMemory(Memory):
    """
    Used for multi-agent games
    """
    def __init__(self, capacity: int):
        super(SettlementMemory, self).__init__(capacity)

    def push(self, *args):
        self.memory.append(Memory.TwoAgentFullTransition(*args))

class ReplayBuffer(object):
    """
    A replay buffer using by MC and Q-Network to store transition
    ----------
    Args:
        capacity: the capacit of replay buffer (int)
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Memory.Transition(*args))
    def clean(self):
        self.memory = deque([],maxlen=self.capacity)
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

    # def sample(self, batch_size: int):
#         LENGTH = 1000
#         if self.__len__() <= LENGTH:
#             return random.sample(self.memory, batch_size)
#         else:
#             return random.sample(list(self.memory)[-LENGTH:], batch_size)