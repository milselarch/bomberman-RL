from typing import List, Optional

import numpy as np

from Transition import Transition, TransitionsBatch


class PrioritizedReplayBuffer:
    def __init__(
        self, capacity: int, epsilon=1e-6, alpha=0.8, beta=0.4,
        beta_increment=0.001
    ):
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha   # how much prioritisation is used
        self.beta = beta    # for importance sampling weights
        self.beta_increment = beta_increment
        self.priority_buffer = np.zeros(self.capacity)
        self.data: List[Transition] = []
        self.position = 0

    def length(self):
        return len(self.data)

    def push(self, transition: Transition, priority: Optional[float] = None):
        max_priority = np.max(self.priority_buffer) if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition

        if priority is None:
            priority = max_priority

        self.priority_buffer[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priority_buffer[:len(self.data)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            len(self.data), batch_size, p=probabilities
        )
        experiences: List[Transition] = [self.data[i] for i in indices]

        total = len(self.data)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = np.min([1., self.beta + self.beta_increment])
        batch = TransitionsBatch.build(experiences)
        return batch

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priority_buffer[idx] = error + self.epsilon