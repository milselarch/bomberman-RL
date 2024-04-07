import dataclasses


@dataclasses.dataclass
class Incentives(object):
    IN_BOMB_RANGE_PENALTY: float = 0
    NOT_IN_BOMB_RANGE_PENALTY: float = 0
    MOVING_INTO_BOMB_RANGE_PENALTY: float = 0
    MOVING_FROM_BOMB_RANGE_REWARD: float = 0
    NOT_MOVING_FROM_BOMB_RANGE_PENALTY: float = 0
    WAITING_BESIDE_BOMB_RANGE_REWARD: float = 0
    TRYING_TO_ENTER_WALL_PENALTY: float = 0
    BOXES_IN_BOMB_RANGE_REWARD: float = 0
    WALK_INTO_SPACE_REWARD: float = 0
    SAME_GRID_PENALTY: float = 0
    TRAPPED_WITH_BOMB_PENALTY: float = 0
    DEATH_PENALTY: float = 0

    DESTROY_BOX_REWARD: float = 0
    DESTROY_ENEMY_REWARD: float = 0

    TRAPPED_THEMSELVES_GUARANTEED_DEATH: float = 0
    STAY_ALIVE: float = 0
