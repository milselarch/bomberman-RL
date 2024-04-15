from abc import ABC, abstractmethod


class Actor(ABC):
    def __init__(self):
        self.bomb_limit: int = 1

    @abstractmethod
    def is_player(self) -> bool:
        raise NotImplementedError