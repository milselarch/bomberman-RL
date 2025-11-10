from enum import Enum


class Algorithm(Enum):
    DFS = 0
    DIJKSTRA = 1
    PLAYER = 2
    # enemy does not exist
    NONE = 3
    # enemy exists but does not move
    STILL = 4
