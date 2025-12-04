import dataclasses


@dataclasses.dataclass
class Settings(object):
    num_episodes: int = 500

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    BATCH_SIZE = 128
    # GAMMA is the discount factor as mentioned in the previous section
    GAMMA = 0.99
    # EPS_START is the starting value of epsilon
    EPS_START = 0.9
    # EPS_END is the final value of epsilon
    EPS_END = 0.01
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    EPS_DECAY = 2500
    # TAU is the update rate of the target network
    TAU = 0.005
    # LR is the learning rate of the ``AdamW`` optimizer
    LR = 3e-4

