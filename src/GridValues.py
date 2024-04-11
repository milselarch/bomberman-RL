from enum import IntEnum


class GridValues(IntEnum):
    EMPTY_GRID_VAL = 0
    WALL_GRID_VAL = 1
    BOX_GRID_VAL = 2
    BOMB_GRID_VAL = 3
    BOMB_FRAME_2_GRID_VAL = 20
    BOMB_FRAME_1_GRID_VAL = 21
    EXPLOSION_GRID_VAL = 9
    ENEMY_GRID_VAL = 4
    PLAYER_GRID_VAL = 5
    # 1 is for indestructible walls
    # 2 is for destructible walls
    # 3 is for bombs
    # 4 for enemies
    # 5 for player
    # 4+3=7 for enemy dropping bomb
    # 5+3=8 for player dropping bomb
    # 9 for explosion.
    # 9+2 = 11 for explosion with box
    # 9+4 = 13 for explosion with enemy
    # 9+5 = 14 for explosion with player
    # 20 for bomb frame 2 (going to explode)
    # 21 for bomb frame 1 (right before exploding)
    
