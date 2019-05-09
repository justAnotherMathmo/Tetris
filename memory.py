import random
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_piece'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        fleeting_memory = Transition(*args)
        self.memory[self.position] = fleeting_memory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def dump(self, path):
        """Dumps the state of the memory to disk"""

    def __len__(self):
        return len(self.memory)


class BiasedMemory(ReplayMemory):
    def __init__(self, capacity, bias_threshold_fraction):
        super().__init__(capacity)
        self.bias = []
        self.bias_sum = 0
        self.bias_threshold_fraction = bias_threshold_fraction

    def push(self, *args, bias=1):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.bias.append(None)
            self.bias_sum += bias
        else:
            # Don't add if small bias
            if bias < self.bias_sum / len(self.memory) * self.bias_threshold_fraction:
                return
            self.bias_sum -= self.bias[self.position]
            self.bias_sum += bias
        self.memory[self.position] = Transition(*args)
        self.bias[self.position] = bias
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, biased=True):
        if biased:
            choice_indices = np.random.choice(
                len(self.memory), size=batch_size, replace=False, p=np.array(self.bias) / self.bias_sum
            )
            return [self.memory[i] for i in choice_indices]
        else:
            return random.sample(self.memory, batch_size)
