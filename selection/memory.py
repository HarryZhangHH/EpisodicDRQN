from collections import namedtuple, deque
import random

class Memory(object):
    """
    Used for multi-agent games
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        pass
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def clean(self):
        self.memory = deque([],maxlen=self.capacity)
    def __len__(self):
        return len(self.memory)