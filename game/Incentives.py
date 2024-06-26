import dataclasses


@dataclasses.dataclass
class Incentives(object):
    PUT_BOMB_HAVE_ESCAPE_ROUTE_REWARD: float = 0
    PUT_BOMB_NO_ESCAPE_ROUTE_PENALTY: float = 0
    IN_BOMB_RANGE_PENALTY: float = 0
    NOT_IN_BOMB_RANGE_REWARD: float = 0
    MOVING_INTO_BOMB_RANGE_PENALTY: float = 0
    MOVING_FROM_BOMB_RANGE_REWARD: float = 0
    NOT_MOVING_FROM_BOMB_RANGE_PENALTY: float = 0
    WAITING_BESIDE_BOMB_RANGE_REWARD: float = 0
    WAITING_BESIDE_EXPLOSION_REWARD: float = 0
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

    BOX_GRAVITY: float = 0
    ENEMY_GRAVITY: float = 0
    TARGET_ENEMY_GRAVITY: float = 0
    BOMB_GRAVITY: float = 0
    BOMB_DEATH_DISTANCE_PENALTY: float = 0

    PLACE_BOMB_CONSIDER_BOX_COUNT: float = 0
    PLACE_BOMB_CONSIDER_ENEMY_COUNT: float = 0
    PLACE_BOMB_CONSIDER_ENEMY_COUNT_INF_RANGE: float = 0
    PLACE_BOMB_CONSIDER_ENEMY_COUNT_MISS_BY_ONE: float = 0
    PLACE_BOMB_CONSIDER_ENEMY_COUNT_IN_SQUARE: float = 0
    BOMB_DEATH_DISTANCE_PENALTY: float = 0
    # penalizes placing the first bomb in the spawn area
    FIRST_CORNER_BOMB_PENALTY: float = 0
