from __future__ import annotations

import numpy as np

from typing import List
from dataclasses import dataclass, field


@dataclass
class Transition(object):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class TransitionsBatch(object):
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)

    @staticmethod
    def build(transitions: List[Transition]) -> TransitionsBatch:
        batch = TransitionsBatch()
        for transition in transitions:
            batch.states.append(transition.state)
            batch.actions.append(transition.action)
            batch.rewards.append(transition.reward)
            batch.next_states.append(transition.next_state)
            batch.dones.append(transition.done)

        return batch