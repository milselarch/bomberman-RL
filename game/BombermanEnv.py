import random
import time
import numpy as np
import pygame
# import matplotlib.pyplot as plt

from typing import List, Tuple
from game.GridValues import GridValues
from game.Incentives import Incentives
from enums.algorithm import Algorithm
from game.player import Player
from game.enemy import Enemy
from game.explosion import Explosion
from game.bomb import Bomb

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
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

GRID_BASE = np.array(GRID_BASE_LIST)


class BombermanEnv(object):
    BACKGROUND_COLOR = (107, 142, 35)

    def __init__(
        self, surface, path, player_alg, en1_alg, en2_alg,
        en3_alg, scale, physics_fps: int = 15, render_fps: int = 15,
        simulate_time: bool = False, incentives: Incentives = Incentives(),
        max_steps: int = 3000
    ):
        """
        :param surface:
        :param path:
        :param player_alg:
        :param en1_alg:
        :param en2_alg:
        :param en3_alg:
        :param scale:
        :param physics_fps: physics update rate
        :param render_fps: game UI update rate
        :param simulate_time:
        whether to simulate the passage of time between physics updates
        or actually wait between physics updates to match physics_fps
        setting simulate_time should simulate the game faster
        :param incentives:
        """
        self.incentives = incentives
        self.physics_fps = physics_fps
        self.render_fps = render_fps
        self.simulate_time = simulate_time
        self.max_steps = max_steps

        # cumulative sum of rewards recieved throughout the game
        self._score: float = 0.0
        self.game_time_passed: int = 0  # time passed in milliseconds
        self.last_update_stamp: float = -float('inf')

        self.surface = surface
        self.path = path
        self.scale = scale

        # self.grid = [row[:] for row in GRID_BASE]
        self.grid = GRID_BASE.copy()
        self.generate_map()
        # self.grid = np.array(self.grid)
        self.grid_state = self.grid.copy()
        self.player_boxes_destroyed = 0
        self.player_kills = 0
        self._steps = 0

        self.MAX_VAL_IN_GRID = GridValues.EXPLOSION_GRID_VAL

        # Height of whole map, i.e. no. of rows
        self.m = len(self.grid)
        #  Width of whole map, i.e. no. of columns/fields in each row
        self.n = len(self.grid[0])
        self.state_shape = (8, self.m, self.n, 1)
        self.UP = 'U'
        self.DOWN = 'D'
        self.LEFT = 'L'
        self.RIGHT = 'R'
        self.BOMB = 'Bomb'
        self.WAIT = 'Wait'

        self.action_space = [
            self.UP, self.DOWN, self.LEFT, self.RIGHT,
            self.BOMB, self.WAIT
        ]

        self.action_space_size = len(self.action_space)
        self.actions_shape = (self.action_space_size,)
        self.clock = pygame.time.Clock()
        self._game_ended = False

        # self.font = pygame.font.SysFont('Bebas', scale)
        pygame.init()
        self.font = pygame.font.SysFont('Arial', scale)

        self.explosions: List[Explosion] = []
        self.bombs: List[Bomb] = []
        self.power_ups = []
        self.bombs.clear()
        self.explosions.clear()
        self.power_ups.clear()

        self.enemy_list = []
        self.enemy_blocks = []
        self.enemies_prev_grid_pos_x = []
        self.enemies_prev_grid_pos_y = []

        self.player = Player()
        self.player_prev_grid_pos_x = self.player.grid_x
        self.player_prev_grid_pos_y = self.player.grid_y
        self.player_direction_x = 0
        self.player_direction_y = 0
        self.player_moving = False
        self.player_moving_action = ''
        self.current_player_direction = 0
        self.player_next_grid_pos_x = None
        self.player_next_grid_pos_y = None
        self.player_in_bomb_range = False

        self.steps_player_in_same_pos = 0
        # self.destGridSqX = self.player.pos_x
        # self.destGridSqY = self.player.pos_y
        # self.hasNoDestinationGrid = True
        # self.toDestGridAction = ''

        self.en1_alg = en1_alg
        self.en2_alg = en2_alg
        self.en3_alg = en3_alg
        self.player_alg = player_alg

        if self.en1_alg is not Algorithm.NONE:
            en1 = Enemy(11, 11, self.en1_alg)
            en1.load_animations('1', self.scale)
            self.enemy_list.append(en1)
            self.enemy_blocks.append(en1)
            self.enemies_prev_grid_pos_x.append(int(en1.pos_x / Enemy.TILE_SIZE))
            self.enemies_prev_grid_pos_y.append(int(en1.pos_y / Enemy.TILE_SIZE))

        if self.en2_alg is not Algorithm.NONE:
            en2 = Enemy(1, 11, self.en2_alg)
            en2.load_animations('2', self.scale)
            self.enemy_list.append(en2)
            self.enemy_blocks.append(en2)
            self.enemies_prev_grid_pos_x.append(int(en2.pos_x / Enemy.TILE_SIZE))
            self.enemies_prev_grid_pos_y.append(int(en2.pos_y / Enemy.TILE_SIZE))

        if self.en3_alg is not Algorithm.NONE:
            en3 = Enemy(11, 1, self.en3_alg)
            en3.load_animations('3', self.scale)
            self.enemy_list.append(en3)
            self.enemy_blocks.append(en3)
            self.enemies_prev_grid_pos_x.append(int(en3.pos_x / Enemy.TILE_SIZE))
            self.enemies_prev_grid_pos_y.append(int(en3.pos_y / Enemy.TILE_SIZE))

        if self.player_alg is Algorithm.PLAYER:
            self.player.load_animations(self.scale)
            self.enemy_blocks.append(self.player)

        # elif self.player_alg is not Algorithm.NONE:
        #     en0 = Enemy(1, 1, self.player_alg)
        #     en0.load_animations('', self.scale)
        #     self.enemyList.append(en0)
        #     self.enemyBlocks.append(en0)
        #     self.player.life = False
        else:
            self.player.life = False

        self.set_player_in_grid()
        self.set_enemies_in_grid()

        transform = pygame.transform
        grass_img = pygame.image.load('images/terrain/grass.png')
        grass_img = pygame.transform.scale(grass_img, (scale, scale))
        block_img = pygame.image.load('images/terrain/block.png')
        block_img = pygame.transform.scale(block_img, (scale, scale))
        box_img = pygame.image.load('images/terrain/box.png')
        box_img = pygame.transform.scale(box_img, (scale, scale))
        bomb1_img = pygame.image.load('images/bomb/1.png')
        bomb1_img = pygame.transform.scale(bomb1_img, (scale, scale))
        bomb2_img = pygame.image.load('images/bomb/2.png')
        bomb2_img = pygame.transform.scale(bomb2_img, (scale, scale))
        bomb3_img = pygame.image.load('images/bomb/3.png')
        bomb3_img = pygame.transform.scale(bomb3_img, (scale, scale))

        explosion1_img = pygame.image.load('images/explosion/1.png')
        explosion1_img = transform.scale(explosion1_img, (scale, scale))
        explosion2_img = pygame.image.load('images/explosion/2.png')
        explosion2_img = transform.scale(explosion2_img, (scale, scale))
        explosion3_img = pygame.image.load('images/explosion/3.png')
        explosion3_img = transform.scale(explosion3_img, (scale, scale))

        self.terrain_images = [grass_img, block_img, box_img, grass_img]
        self.bombImages = [bomb1_img, bomb2_img, bomb3_img]
        self.explosionImages = [explosion1_img, explosion2_img, explosion3_img]

        power_up_bomb_img = pygame.image.load('images/power_up/bomb.png')
        power_up_bomb_img = transform.scale(power_up_bomb_img, (scale, scale))
        power_up_fire_img = pygame.image.load('images/power_up/fire.png')
        power_up_fire_img = transform.scale(power_up_fire_img, (scale, scale))
        self.power_ups_images = [power_up_bomb_img, power_up_fire_img]

    def get_score(self) -> float:
        return self._score

    def draw(self):
        ############################################
        ### FOR RENDERING THE GAME IN THE WINDOW ###
        ############################################
        self.surface.fill(self.BACKGROUND_COLOR)

        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                self.surface.blit(
                    self.terrain_images[self.grid[i][j]],
                    (i * self.scale, j * self.scale, self.scale, self.scale)
                )

        for pu in self.power_ups:
            self.surface.blit(self.power_ups_images[pu.type.value], (
                pu.pos_x * self.scale, pu.pos_y * self.scale,
                self.scale, self.scale
            ))
        for x in self.bombs:
            self.surface.blit(self.bombImages[x.frame], (
                x.pos_x * self.scale, x.pos_y * self.scale,
                self.scale, self.scale
            ))
        for y in self.explosions:
            for x in y.sectors:
                self.surface.blit(self.explosionImages[y.frame], (
                    x[0] * self.scale, x[1] * self.scale,
                    self.scale, self.scale
                ))

        if self.player.life:
            direction = self.player.direction
            self.surface.blit(
                self.player.animation[direction][self.player.frame], (
                    self.player.pos_x * (self.scale / 4),
                    self.player.pos_y * (self.scale / 4), self.scale,
                    self.scale
                )
            )

        for en in self.enemy_list:
            if not en.life:
                continue

            self.surface.blit(en.animation[en.direction][en.frame], (
                en.pos_x * (self.scale / 4), en.pos_y * (self.scale / 4),
                self.scale, self.scale
            ))

            if self.path:
                if en.algorithm == Algorithm.DFS:
                    for sek in en.path:
                        pygame.draw.rect(
                            self.surface, (255, 0, 0, 240),
                            [sek[0] * self.scale, sek[1] * self.scale,
                            self.scale, self.scale], 1
                        )
                else:
                    for sek in en.path:
                        pygame.draw.rect(self.surface, (255, 0, 255, 240), [
                            sek[0] * self.scale, sek[1] * self.scale,
                            self.scale, self.scale
                        ], 1)

        if self._game_ended:
            tf = self.font.render(
                "Press ESC to go back to menu",
                False, (153, 153, 255)
            )
            self.surface.blit(tf, (10, 10))

        pygame.display.update()

    def generate_map(self):
        ####################################################################
        """ This is just generating destroyable boxes if I am not wrong. """
        ####################################################################

        for i in range(1, len(self.grid) - 1):
            for j in range(1, len(self.grid[i]) - 1):
                if self.grid[i][j] != 0:
                    continue

                out_of_grid = (
                    (i < 3 or i > len(self.grid) - 4) and
                    (j < 3 or j > len(self.grid[i]) - 4)
                )
                if out_of_grid:
                    continue

                if random.randint(0, 9) < 7:
                    self.grid[i][j] = GridValues.BOX_GRID_VAL
                    pass

        return

    def set_enemies_in_grid(self):
        for i in range(len(self.enemy_list)):
            x = self.enemies_prev_grid_pos_x[i]
            y = self.enemies_prev_grid_pos_y[i]
            self.grid_state[x][y] = GridValues.EMPTY_GRID_VAL
            if self.grid_state[x][y] < 0:
                self.grid_state[x][y] = 0

            self.enemies_prev_grid_pos_x[i] = int(
                self.enemy_list[i].pos_x / Enemy.TILE_SIZE
            )
            self.enemies_prev_grid_pos_y[i] = int(
                self.enemy_list[i].pos_y / Enemy.TILE_SIZE
            )

            assert self.grid_state[x][y] == GridValues.EMPTY_GRID_VAL
            self.grid_state[x][y] = GridValues.ENEMY_GRID_VAL

    def clear_enemy_from_grid(self, enemy):
        x = (int(enemy.pos_x / Enemy.TILE_SIZE))
        y = (int(enemy.pos_y / Enemy.TILE_SIZE))
        self.grid_state[x][y] = GridValues.EMPTY_GRID_VAL

    def set_player_in_grid(self):
        x = self.player_prev_grid_pos_x
        y = self.player_prev_grid_pos_y
        self.grid_state[x][y] = GridValues.EMPTY_GRID_VAL

        tile_size = Player.TILE_SIZE
        self.player_prev_grid_pos_x = int(self.player.pos_x / tile_size)
        self.player_prev_grid_pos_y = int(self.player.pos_y / tile_size)

        x = self.player_prev_grid_pos_x
        y = self.player_prev_grid_pos_y
        self.grid_state[x][y] = GridValues.PLAYER_GRID_VAL

    def clear_player_from_grid(self):
        x = self.player_prev_grid_pos_x
        y = self.player_prev_grid_pos_y
        self.grid_state[x][y] = GridValues.EMPTY_GRID_VAL

    def set_explosions_in_grid(self) -> int:
        player_destroyed_boxes = 0

        for i in range(len(self.explosions)):
            explosion = self.explosions[i]

            for grid_coord in explosion.sectors:
                x, y = grid_coord

                is_player_destroyed_box = (
                    explosion.bomber.is_player() and
                    (self.grid_state[x][y] == GridValues.BOX_GRID_VAL)
                )

                if is_player_destroyed_box:
                    player_destroyed_boxes += 1

                self.grid_state[x][y] = GridValues.EXPLOSION_GRID_VAL

        return player_destroyed_boxes

    def clear_explosion_from_grid(self, explosion_obj):
        for grid_coords_tuple in explosion_obj.sectors:
            # Set to 0 as nothing should be left if the explosion
            # occurred on the grid square
            self.grid_state[grid_coords_tuple[0]][grid_coords_tuple[1]] = 0

    def is_game_ended(self) -> bool:
        if not self.player.life:
            return True
        if self._steps >= self.max_steps:
            return True

        for en in self.enemy_list:
            if en.life:
                return False

        return True

    def update_bombs(self, dt) -> Tuple[int, int]:
        """
        :param dt:
        :return:
        number of player kills
        """
        player_kills, player_destroyed_boxes = 0, 0

        for bomb in self.bombs:
            x, y = bomb.pos_x, bomb.pos_y
            bomb.update(dt)

            if bomb.time < 1:
                bomb.bomber.bomb_limit += 1
                self.grid[x][y] = 0

                explosion = Explosion(x, y, bomb.range)
                self.explosions.append(explosion)
                explosion.explode(self.grid, self.bombs, bomb, self.power_ups)
                destroyed_boxes = explosion.clear_sectors(
                    self.grid, np.random, self.power_ups
                )

                if explosion.bomber.is_player():
                    player_destroyed_boxes += destroyed_boxes

            elif bomb.time < 5:
                self.set_explosions_in_grid()

        if self.player not in self.enemy_list:
            self.player.check_death(self.explosions)
            if not self.player.life:
                self.clear_player_from_grid()

        for enemy in self.enemy_list:
            _, killed_by_player = enemy.check_death(self.explosions)

            if killed_by_player:
                assert not enemy.life
                player_kills += 1

            if not enemy.life:
                self.clear_enemy_from_grid(enemy)

        for explosion in self.explosions:
            explosion.update(dt)
            if explosion.time < 1:
                self.explosions.remove(explosion)
                self.clear_explosion_from_grid(explosion)

        return player_kills, player_destroyed_boxes

    def check_if_in_bomb_range(self):
        player_pos_x = self.player.pos_x
        player_pos_y = self.player.pos_y

        for bomb in self.bombs:
            """
            bomb.sectors array stores all positions that
            the bomb explosion would hit.
            """
            for explosion_field_coords in bomb.sectors:
                x, y = explosion_field_coords
                in_bomb_range = (
                    (int(player_pos_x / Player.TILE_SIZE) == x) and
                    (int(player_pos_y / Player.TILE_SIZE) == y)
                )

                if in_bomb_range:
                    return True

        return False

    def check_if_walking_to_bomb_range(self):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y

        if not self.player_in_bomb_range:
            for bomb in self.bombs:
                """
                bomb.sectors array stores all positions that the bomb
                explosion would hit.
                """
                for explosionFieldCoords in bomb.sectors:
                    x, y = explosionFieldCoords
                    in_bomb_range = (
                        int(playerPosX / Player.TILE_SIZE) == x and
                        int(playerPosY / Player.TILE_SIZE) == y
                    )

                    if in_bomb_range:
                        self.player_in_bomb_range = True
                        return True

        # If player is not walking into bomb range,
        # or is already in bomb range, return False
        return False

    def check_if_walking_out_of_bomb_range(self):
        player_pos_x = self.player.pos_x
        player_pos_y = self.player.pos_y

        if self.player_in_bomb_range:
            for bomb in self.bombs:
                """
                bomb.sectors array stores all positions that the 
                bomb explosion would hit. 
                """
                for explosion_field_coords in bomb.sectors:
                    x, y = explosion_field_coords
                    in_explosion_range = (
                        int(player_pos_x / Player.TILE_SIZE) == x and
                        int(player_pos_y / Player.TILE_SIZE) == y
                    )
                    if in_explosion_range:
                        # As long as player's grid is still in any
                        # explosion range, return false.
                        return False

            # If player's grid is not in any explosion range
            # and player was originally in bomb range, return true.
            self.player_in_bomb_range = False
            return True

        # If player is previously and currently not in bomb range, return False
        return False

    def check_if_waiting_beside_bomb_range(self, action):
        grid_x = int(self.player.pos_x / Player.TILE_SIZE)
        grid_y = int(self.player.pos_y / Player.TILE_SIZE)
        top_pos = (grid_x, grid_y - 1)
        bottom_pos = (grid_x, grid_y + 1)
        left_pos = (grid_x - 1, grid_y)
        right_pos = (grid_x + 1, grid_y)

        if not self.player_in_bomb_range and action == self.WAIT:
            for bomb in self.bombs:
                """
                bomb.sectors array stores all positions that
                 the bomb explosion would hit.
                """
                for explosion_field_coords in bomb.sectors:
                    x, y = explosion_field_coords
                    in_explosion_range = (
                        (top_pos[0] == x and top_pos[1] == y) or
                        (bottom_pos[0] == x and bottom_pos[1] == y) or
                        (left_pos[0] == x and left_pos[1] == y) or
                        (right_pos[0] == x and right_pos[1] == y)
                    )
                    if in_explosion_range:
                        # If top, bottom, left or right grid of player's
                        # grid is in any explosion range, return True.
                        return True

        return False

    def check_if_own_bomb_to_hit_boxes(self, player_bomb):
        # Only give reward when player just planted bomb
        player_grid_pos_x = int(self.player.pos_x / Player.TILE_SIZE)
        player_grid_pos_y = int(self.player.pos_y / Player.TILE_SIZE)
        at_bomb = (
            player_grid_pos_x == player_bomb.pos_x and
            player_grid_pos_y == player_bomb.pos_y
        )

        if at_bomb:
            for explosion_fields_coords in player_bomb.sectors:
                x, y = explosion_fields_coords
                if self.grid[x][y] == 2:
                    # 2 in grid means the field contains a destructible box.
                    return True

        return False

    def check_if_own_bomb_to_hit_enemy(self, player_bomb):
        for explosion_fields_coords in player_bomb.sectors:
            x, y = explosion_fields_coords

            for enemy in self.enemy_list:
                if enemy.pos_x == x and enemy.pos_y == y:
                    return True

        return False

    def check_if_walk_into_obj(self, action):
        player_pos_x = self.player.pos_x
        player_pos_y = self.player.pos_y
        x = 0
        y = 0

        if action == self.DOWN:
            y = 1
        elif action == self.RIGHT:
            x = 1
        elif action == self.UP:
            y = -1
        elif action == self.LEFT:
            x = -1

        grid_x = int(player_pos_x / Player.TILE_SIZE)
        grid_y = int(player_pos_y / Player.TILE_SIZE)
        grid_val = self.grid[grid_x + x][grid_y + y]
        return grid_val == 1 or grid_val == 2 or grid_val == 3

    def check_if_trapped_with_bomb(self):
        player_pos_x = self.player.pos_x
        player_pos_y = self.player.pos_y

        grid_x = int(player_pos_x / Player.TILE_SIZE)
        grid_y = int(player_pos_y / Player.TILE_SIZE)
        top = self.grid[grid_x][grid_y-1]
        bottom = self.grid[grid_x][grid_y+1]
        left = self.grid[grid_x+1][grid_y]
        right = self.grid[grid_x-1][grid_y]

        return (
            (top == 1 or top == 2 or top == 3) and
            (bottom == 1 or bottom == 2 or bottom == 3) and
            (left == 1 or left == 2 or left == 3) and
            (right == 1 or right == 2 or right == 3)
        )

    def check_if_walkable_space(self, action):
        player_pos_x = self.player.pos_x
        player_pos_y = self.player.pos_y
        x = 0
        y = 0

        if action == self.DOWN:
            y = 1
        elif action == self.RIGHT:
            x = 1
        elif action == self.UP:
            y = -1
        elif action == self.LEFT:
            x = -1

        grid_x = int(player_pos_x / Player.TILE_SIZE) + x
        grid_y = int(player_pos_y / Player.TILE_SIZE) + y
        return self.grid[grid_x][grid_y] == GridValues.EMPTY_GRID_VAL

    @property
    def steps(self):
        return self._steps

    def step(self, action):
        self._steps += 1
        # print('TICK_FPS', self.tick_fps)
        if self.simulate_time:
            dt = 1000 // self.physics_fps
        else:
            dt = self.clock.tick(self.physics_fps)

        self.game_time_passed += dt
        tile_size = Player.TILE_SIZE

        for enemy in self.enemy_list:
            enemy.make_move(
                self.grid, self.bombs, self.explosions, self.enemy_blocks
            )

        if self.player.life:
            assign_action = False
            self.current_player_direction = self.player.direction
            self.player_moving = True

            self.player_direction_x = 0
            self.player_direction_y = 0
            self.player_moving_action = action

            if action == self.DOWN:
                self.current_player_direction = 0
                self.player_direction_x = 0
                self.player_direction_y = 1
            elif action == self.RIGHT:
                self.current_player_direction = 1
                self.player_direction_x = 1
                self.player_direction_y = 0
            elif action == self.UP:
                self.current_player_direction = 2
                self.player_direction_x = 0
                self.player_direction_y = -1
            elif action == self.LEFT:
                self.current_player_direction = 3
                self.player_direction_x = -1
                self.player_direction_y = 0
            elif action == self.WAIT or action == self.BOMB:
                self.player_direction_x = 0
                self.player_direction_y = 0
                self.player_moving = False
                self.player_moving_action = ''

            # print('ACT', self.player_moving_action, [action])
            if self.player_moving:
                # Storing Destination Grid Coordinates in Grid
                self.player_next_grid_pos_x = int(
                    self.player.pos_x / Player.TILE_SIZE
                ) + self.player_direction_x
                self.player_next_grid_pos_y = int(
                    self.player.pos_y / Player.TILE_SIZE
                ) + self.player_direction_y

                x = self.player_next_grid_pos_x
                y = self.player_next_grid_pos_y
                grid_val = self.grid[x][y]

                if (grid_val == 1) or (grid_val == 2) or (grid_val == 3):
                    # If Destination Grid is a Wall, Destructible Box
                    # or Bomb, Reset Values to not Force Player to Move
                    # in that Direction.
                    self.player_direction_x = 0
                    self.player_direction_y = 0
                    self.player_next_grid_pos_x = None
                    self.player_next_grid_pos_y = None
                    self.player_moving = False
                    self.player_moving_action = ''

            player_next_x = self.player_next_grid_pos_x
            player_next_y = self.player_next_grid_pos_y
            at_destination = (
                self.player_moving and
                int(self.player.pos_x / tile_size) == player_next_x and
                int(self.player.pos_y / tile_size) == player_next_y and
                self.player.pos_x % tile_size == 0 and
                self.player.pos_y % tile_size == 0
            )

            if at_destination:
                # If current grid coordinates of player
                # is same as destination grid coordinates,
                # and position of player are multiples of Player.TILE_SIZE,
                # THEN reset values
                assign_action = False
                self.player_direction_x = 0
                self.player_direction_y = 0
                self.player_next_grid_pos_x = None
                self.player_next_grid_pos_y = None
                self.player_moving = False
                self.player_moving_action = ''

            if assign_action:
                action = self.player_moving_action

            # Move player
            self.player.move(
                self.player_direction_x, self.player_direction_y,
                self.grid, self.enemy_blocks, self.power_ups
            )

            if self.current_player_direction != self.player.direction:
                self.player.frame = 0
                self.player.direction = self.current_player_direction

            if self.player_moving:
                if self.player.frame == 2:
                    self.player.frame = 0
                else:
                    self.player.frame += 1

            self.set_enemies_in_grid()
            self.set_player_in_grid()
            # ("POS", self.player.pos_x, self.player.pos_y)

        ############################################
        """ FOR RENDERING THE GAME IN THE WINDOW """
        ############################################

        timestamp = time.time()
        time_since_last_render = timestamp - self.last_update_stamp
        update_interval = 1.0 / self.render_fps

        if time_since_last_render >= update_interval:
            self.last_update_stamp = timestamp
            pygame.event.get()
            self.draw()

        has_dropped_bomb = False
        player_bomb = None

        if action == self.BOMB:
            if self.player.bomb_limit != 0 and self.player.life:
                has_dropped_bomb = True
                player_bomb = self.player.plant_bomb(self.grid)
                self.bombs.append(player_bomb)
                x = player_bomb.pos_x
                y = player_bomb.pos_y

                self.grid[x][y] = GridValues.BOMB_GRID_VAL
                self.player.bomb_limit -= 1

        I: Incentives = self.incentives
        reward: float = 0

        player_kills, player_destroyed_boxes = self.update_bombs(dt)
        self.player_boxes_destroyed += player_destroyed_boxes
        self.player_kills += player_kills

        destroy_enemy_reward = I.DESTROY_ENEMY_REWARD * player_kills
        destroy_box_reward = I.DESTROY_BOX_REWARD * player_destroyed_boxes
        reward += destroy_enemy_reward + destroy_box_reward

        """
        if (destroy_box_reward != 0) or (destroy_enemy_reward != 0):
            print("DESTROY", destroy_box_reward, destroy_enemy_reward)
        """

        if not self._game_ended:
            self._game_ended = self.is_game_ended()

        ######################################
        """ REWARDS AND PENALTIES SECTION """
        ######################################

        """ 
        Very high positive and negative rewards to prevent AI 
        from only moving a little in a direction before changing directions.
        """
        # NOT_MOVING_TO_DEST_GRID_PENALTY = -1000
        # MOVING_TO_DEST_GRID_PENALTY = 1000

        if self.check_if_in_bomb_range():
            reward += I.IN_BOMB_RANGE_PENALTY
        else:
            reward += I.NOT_IN_BOMB_RANGE_PENALTY

        if self.check_if_walking_to_bomb_range():
            reward += I.MOVING_INTO_BOMB_RANGE_PENALTY

        if self.check_if_walking_out_of_bomb_range():
            reward += I.MOVING_FROM_BOMB_RANGE_REWARD
        else:
            reward += I.NOT_MOVING_FROM_BOMB_RANGE_PENALTY

        if self.check_if_waiting_beside_bomb_range(action):
            reward += I.WAITING_BESIDE_BOMB_RANGE_REWARD

        if has_dropped_bomb and self.check_if_own_bomb_to_hit_boxes(player_bomb):
            reward += I.BOXES_IN_BOMB_RANGE_REWARD

        if self.check_if_walk_into_obj(action):
            reward += I.TRYING_TO_ENTER_WALL_PENALTY

        if self.check_if_trapped_with_bomb():
            reward += I.TRAPPED_WITH_BOMB_PENALTY

        if not self.player.life:
            reward += I.DEATH_PENALTY
            # print('DIE', I.DEATH_PENALTY)
            self.clear_player_from_grid()

        self._score += reward
        return (
            self.get_normalised_state(), reward,
            self.is_game_ended(), self.player_moving
        )

    def get_normalised_state(self):
        raw_state = np.array(self.grid_state)
        one_hot_states = []

        for grid_value in GridValues:
            grid_value = int(grid_value)
            one_hot = np.zeros_like(raw_state).astype(np.float32)
            one_hot[np.where(raw_state == grid_value)] = 1.0
            one_hot_states.append(one_hot)

            if grid_value == GridValues.PLAYER_GRID_VAL:
                pass

        bomb_waits = np.zeros_like(raw_state).astype(np.float32)
        for bomb in self.bombs:
            x, y = bomb.pos_x, bomb.pos_y
            bomb_waits[x][y] = (
                bomb.time_waited / Bomb.WAIT_DURATION
            )

        one_hot_states.append(bomb_waits)
        xy_state = np.stack(one_hot_states, axis=0)
        mid_x = len(self.grid) // 2
        mid_y = len(self.grid[0]) // 2
        shift_x = mid_x - self.player.grid_x
        shift_y = mid_y - self.player.grid_y

        xy_state = np.roll(xy_state, shift_x, axis=1)
        xy_state = np.roll(xy_state, shift_y, axis=2)
        yx_state = np.transpose(xy_state, axes=(0, 2, 1))
        return yx_state

    def reset(self):
        # self.grid = [row[:] for row in GRID_BASE]
        # self.grid = np.array(GRID_BASE)
        self.grid = GRID_BASE.copy()
        self.generate_map()
        self.grid_state = self.grid.copy()
        self.player_boxes_destroyed = 0
        self.player_kills = 0
        self._steps = 0

        self._score = 0
        self.game_time_passed = 0
        self.last_update_stamp = -float('inf')

        self.explosions.clear()
        self.bombs.clear()
        self.power_ups.clear()

        self.enemy_list.clear()
        self.enemy_blocks.clear()
        self.enemies_prev_grid_pos_x.clear()
        self.enemies_prev_grid_pos_y.clear()

        self.player = Player()
        self.player_prev_grid_pos_x = self.player.grid_x
        self.player_prev_grid_pos_y = self.player.grid_y
        self.player_direction_x = 0
        self.player_direction_y = 0
        self.player_moving = False
        self.current_player_direction = 0
        self.player_next_grid_pos_x = None
        self.player_next_grid_pos_y = None

        # self.destGridSqX = self.player.pos_x
        # self.destGridSqY = self.player.pos_y
        # self.hasNoDestinationGrid = True
        # self.toDestGridAction = ''

        self.clock = pygame.time.Clock()
        self._game_ended = False

        if self.en1_alg is not Algorithm.NONE:
            en1 = Enemy(11, 11, self.en1_alg)
            en1.load_animations('1', self.scale)
            self.enemy_list.append(en1)
            self.enemy_blocks.append(en1)
            self.enemies_prev_grid_pos_x.append(
                int(en1.pos_x / Enemy.TILE_SIZE)
            )
            self.enemies_prev_grid_pos_y.append(
                int(en1.pos_y / Enemy.TILE_SIZE)
            )

        if self.en2_alg is not Algorithm.NONE:
            en2 = Enemy(1, 11, self.en2_alg)
            en2.load_animations('2', self.scale)
            self.enemy_list.append(en2)
            self.enemy_blocks.append(en2)
            self.enemies_prev_grid_pos_x.append(
                int(en2.pos_x / Enemy.TILE_SIZE)
            )
            self.enemies_prev_grid_pos_y.append(
                int(en2.pos_y / Enemy.TILE_SIZE)
            )

        if self.en3_alg is not Algorithm.NONE:
            en3 = Enemy(11, 1, self.en3_alg)
            en3.load_animations('3', self.scale)
            self.enemy_list.append(en3)
            self.enemy_blocks.append(en3)
            self.enemies_prev_grid_pos_x.append(
                int(en3.pos_x / Enemy.TILE_SIZE)
            )
            self.enemies_prev_grid_pos_y.append(
                int(en3.pos_y / Enemy.TILE_SIZE)
            )

        if self.player_alg is Algorithm.PLAYER:
            self.player.load_animations(self.scale)
            self.enemy_blocks.append(self.player)

        # elif self.player_alg is not Algorithm.NONE:
        #     en0 = Enemy(1, 1, self.player_alg)
        #     en0.load_animations('', self.scale)
        #     self.enemyList.append(en0)
        #     self.enemyBlocks.append(en0)
        #     self.player.life = False
        else:
            self.player.life = False

        self.set_player_in_grid()
        self.set_enemies_in_grid()
        return self.get_normalised_state()

    def is_player_alive(self):
        return self.player.life

    def count_player_kills(self) -> int:
        player_kills = 0

        for enemy in self.enemy_list:
            if enemy.killed_by_player:
                player_kills += 1

        return player_kills

    def action_space_sample(self):
        #####################################
        """ Just randomly take any action """
        #####################################
        return np.random.choice(self.action_space)

    @staticmethod
    def max_action(Q, state, actions):
        """
        Make an array of values for all actions at a particular state,
        with each value calculated with:
            Q[state, action] + ALPHA * (
                reward + GAMMA * Q[state, actionOfPrevMaxValue] -
                Q[state, action]
            )
        """

        state = tuple(map(tuple, state))

        for a in actions:
            if (state, a) not in Q:
                Q[state, a] = 0

        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        return actions[action]
