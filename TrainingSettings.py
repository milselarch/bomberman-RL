import dataclasses


@dataclasses.dataclass
class TrainingSettings(object):
    IS_MANUAL_CONTROL: bool = False
    IS_CHECKING_ILLEGAL_ACTION: bool = True
    IS_PRESET_GRID: bool = False
    POOL_TRANSITIONS: bool = True

    learning_rate: float = 0.001
    # exponential decay rate for epsilon-greedy exploration rate
    exploration_decay: float = 0.9995  # 0.95
    # initial exploration rate
    # (ie. what fraction of actions are initially randomly chose)
    exploration_max: float = 0.2
    # minimum exploration rate
    exploration_min: float = 0.001  # 0.01
    # discount factor in Q value estimation equation
    # Q(s) = R + gamma * max(Q(s'))
    gamma: float = 0.9  # 0.975
    # how many episodes between target Q network updates
    update_target_every: int = 10
    episode_buffer_size: int = 256
    # total number of episodes to train for
    episodes: int = 50 * 1000
    # length of pooled transition
    pool_duration: float = 4
    # whether to print debug statements onto console
    verbose: bool = False