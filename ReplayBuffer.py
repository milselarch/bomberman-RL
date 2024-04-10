import random

from collections import deque
from Transition import Transition, TransitionsBatch


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def length(self):
        return len(self.memory)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        batch = TransitionsBatch.build(experiences)
        return batch

    def __len__(self):
        return self.length()