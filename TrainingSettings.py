import dataclasses
from enum import IntEnum


class PresetGrid:
    class PresetGridSelect(IntEnum):
        NO_PRESET = 0
        SELECT_GRID_BASE_LIST = 1
        SELECT_GRID_BASE_LIST_PRESET_BOXES = 2
        SELECT_EMPTY_GRID = 3

    def __init__(self, preset_number=PresetGridSelect.NO_PRESET):
        self.grid = None
        match preset_number:
            case PresetGrid.PresetGridSelect.SELECT_GRID_BASE_LIST:
                self.grid = PresetGrid.GRID_BASE_LIST
            case PresetGrid.PresetGridSelect.SELECT_GRID_BASE_LIST_PRESET_BOXES:
                self.grid = PresetGrid.GRID_BASE_LIST_PRESET_BOXES
            case PresetGrid.PresetGridSelect.SELECT_EMPTY_GRID:
                self.grid = PresetGrid.EMPTY_GRID
            case _:
                pass
        self.is_set = self.grid is not None

    GRID_BASE_LIST = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    GRID_BASE_LIST_PRESET_BOXES = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 0, 1],
        [1, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 0, 1],
        [1, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2, 1],
        [1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1],
        [1, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 1],
        [1, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 2, 1],
        [1, 2, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 1],
        [1, 2, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1],
        [1, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 0, 1, 2, 1, 2, 1, 0, 1, 2, 1, 0, 1],
        [1, 0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    EMPTY_GRID = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]


@dataclasses.dataclass
class TrainingSettings(object):
    IS_MANUAL_CONTROL: bool = False
    IS_CHECKING_ILLEGAL_ACTION: bool = True
    PRESET_GRID: PresetGrid = PresetGrid(PresetGrid.PresetGridSelect.NO_PRESET)
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

    # whether to run the model using GPU
    use_gpu: bool = True
    """
    whether to simulate the passage of time between physics updates
    or actually wait between physics updates to match physics_fps
    setting simulate_time should simulate the game faster
    """
    simulate_time: bool = True
    # physics simulation rate per second
    # ignored if simulate_time is True
    physics_fps: int = 15
    # UI render frame rate per second
    render_fps: int = 15

