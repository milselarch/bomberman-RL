from enum import IntEnum


class GridValues(IntEnum):
    EMPTY_GRID_VAL = 0
    WALL_GRID_VAL = 1
    BOX_GRID_VAL = 2
    BOMB_GRID_VAL = 3
    ENEMY_GRID_VAL = 4
    PLAYER_GRID_VAL = 5
    EXPLOSION_GRID_VAL = 9
    # 1 is for indestructible walls
    # 2 is for destructible boxes
    # 3 is for bombs
    # 4 for enemies
    # 5 for player
    # 9 for explosion.