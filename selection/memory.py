from collections import namedtuple, deque
import random

class Memory(object):
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

Agent = namedtuple('Agent', ['agent_1', 'agent_2', 'action_1', 'action_2', 'reward_1', 'reward_2', 'state_1', 'state_2'])
AgentLog = namedtuple('Agent', ['agent_1', 'agent_2', 'action_1', 'action_2', 'reward_1', 'reward_2', 'state_1', 'state_2','next_state_1', 'next_state_2'])
Buffer = namedtuple('ReplyBuffer', ['state', 'action', 'reward', 'next_state'])

class UpdateMemory(Memory):
    """
    Used for multi-agent games
    """
    def __init__(self, capacity: int):
        super(UpdateMemory, self).__init__(capacity)

    def push(self, *args):
        self.memory.append(Agent(*args))

class ReplayBuffer(Memory):
    """
    Replay Buffer
    """
    def __init__(self, capacity: int):
        super(ReplayBuffer, self).__init__(capacity)

    def push(self, *args):
        self.memory.append(Buffer(*args))

    def sample(self, batch_size: int):
        LENGTH = 1000
        if self.__len__() <= LENGTH:
            return random.sample(self.memory, batch_size)
        else:
            return random.sample(list(self.memory)[-LENGTH:], batch_size)

class SettlementMemory(Memory):
    """
    Used for multi-agent games
    """
    def __init__(self, capacity: int):
        super(SettlementMemory, self).__init__(capacity)

    def push(self, *args):
        self.memory.append(AgentLog(*args))