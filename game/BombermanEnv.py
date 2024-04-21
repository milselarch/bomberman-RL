import random
import time
import numpy as np
import pygame
# import matplotlib.pyplot as plt

from typing import List, Tuple
from dataclasses import dataclass
from game.GridValues import GridValues
from game.Incentives import Incentives
from enums.algorithm import Algorithm
from game.player import Player
from game.enemy import Enemy
from game.explosion import Explosion
from game.bomb import Bomb
from astar import find_path

from TrainingSettings import TrainingSettings, PresetGrid


def manhattan_distance(start_grid_coords, end_grid_coords):
    x_dist = abs(end_grid_coords[0] - start_grid_coords[0])
    y_dist = abs(end_grid_coords[1] - start_grid_coords[1])
    return x_dist + y_dist


GRID_BASE = np.array(PresetGrid(PresetGrid.PresetGridSelect.SELECT_GRID_BASE_LIST).grid)


@dataclass
class UpdateBombsResult(object):
    closeness_to_bomb: float
    player_kills: int
    player_destroyed_boxes: int


class BombermanEnv(object):
    BACKGROUND_COLOR = (107, 142, 35)

    def __init__(
        self, surface, path, player_alg, en1_alg, en2_alg,
        en3_alg, scale, incentives: Incentives = Incentives(),
        training_settings: TrainingSettings = TrainingSettings(),
        max_steps: int = 3000,
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
        self.physics_fps = training_settings.physics_fps
        self.render_fps = training_settings.render_fps
        self.simulate_time = training_settings.simulate_time
        self.training_settings = training_settings
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
        self.action_space_idx_map = {
            self.action_space[k]: k for k in range(len(self.action_space))
        }

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
        self.player_bombs_planted = 0
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

        self.FLAG_checkIfPlayerTrappedThemselves = False
        self.lifeDurationCounter = 0
        self.currentTargetEnemy = None
        self.gridCoordIncentiveTick = 1

    def get_score(self) -> float:
        return self._score

    def to_action(self, action_no: int) -> str:
        return self.action_space[action_no]

    def is_move_action_no(self, action_no: int) -> bool:
        move_actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        return self.to_action(action_no) in move_actions

    def draw(self):
        # FOR RENDERING THE GAME IN THE WINDOW
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

        hardcoded = {
            (3, 7), (5, 4), (3, 10), (5, 7), (9, 5), (5, 10),
            (8, 3), (10, 6), (9, 8), (8, 6), (10, 3), (1, 6),
            (2, 5), (2, 8), (7, 4), (7, 1), (7, 7), (6, 8),
            (4, 5), (3, 9), (4, 8), (5, 9), (9, 1), (8, 5),
            (10, 2), (9, 4), (9, 10), (8, 8), (10, 5), (2, 7),
            (10, 8), (6, 1), (1, 8), (6, 4), (7, 9), (6, 7),
            (7, 6), (4, 7), (4, 4), (8, 4), (9, 9), (8, 7),
            (10, 4), (10, 1), (10, 7), (10, 10), (1, 7), (2, 6),
            (6, 6), (7, 5), (6, 3), (6, 9), (7, 8)
        }

        for x,y in hardcoded:
            self.grid[x][y] = GridValues.BOX_GRID_VAL

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

        if self.training_settings.PRESET_GRID.is_set:
            self.grid = np.array(self.training_settings.PRESET_GRID.grid)

        return

    def set_enemies_in_grid(self):
        for i in range(len(self.enemy_list)):
            enemy = self.enemy_list[i]

            prev_x = self.enemies_prev_grid_pos_x[i]
            prev_y = self.enemies_prev_grid_pos_y[i]
            self.grid_state[prev_x][prev_y] = GridValues.EMPTY_GRID_VAL

            x, y = enemy.grid_x, enemy.grid_y
            self.enemies_prev_grid_pos_x[i] = x
            self.enemies_prev_grid_pos_y[i] = y
            self.grid_state[x][y] = GridValues.ENEMY_GRID_VAL

    def clear_enemy_from_grid(self, enemy: Enemy):
        self.grid_state[enemy.grid_x][enemy.grid_y] = GridValues.EMPTY_GRID_VAL

    def set_player_in_grid(self):
        x = self.player_prev_grid_pos_x
        y = self.player_prev_grid_pos_y
        self.grid_state[x][y] = GridValues.EMPTY_GRID_VAL

        x = self.player.grid_x
        y = self.player.grid_y
        self.player_prev_grid_pos_x = x
        self.player_prev_grid_pos_y = y
        self.grid_state[x][y] = GridValues.PLAYER_GRID_VAL

    def clear_player_from_grid(self):
        x = self.player.grid_x
        y = self.player.grid_y
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

    def update_bombs(self, dt) -> UpdateBombsResult:
        """
        :param dt:
        :return:
        number of player kills
        """
        closeness_to_bomb = 0.0
        player_kills, player_destroyed_boxes = 0, 0

        for bomb in self.bombs:
            x, y = bomb.pos_x, bomb.pos_y
            bomb.update(dt)

            if bomb.time < 1:
                bomb.bomber.bomb_limit += 1
                self.grid[x][y] = 0
                self.grid_state[x][y] = 0

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
            closeness_to_bomb = self.player.check_death(self.explosions)
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

        return UpdateBombsResult(
            closeness_to_bomb=closeness_to_bomb,
            player_kills=player_kills,
            player_destroyed_boxes=player_destroyed_boxes
        )

    def check_escape_route_recursive(
        self, x, y, dist_from_bomb, is_same_x,
        is_same_y, is_increase_x, is_increase_y
    ) -> bool:
        # Use grid and not grid_state here as grid has only bombs,
        # walls and boxes values.
        grid_val = self.grid[x][y]
        if 1 <= grid_val <= 3:
            # Wall is 1, Box is 2, Bomb is 3
            return False

        elif dist_from_bomb > self.player.range:
            # Check if this grid is out of bomb range and
            # is not wall, box or bomb. If so, there might be
            # a possible an escape route
            return True
        else:
            return (
                self.grid[x + 1][y] == 0 or self.grid[x - 1][y] == 0 or
                self.grid[x][y + 1] == 0 or self.grid[x][y - 1] == 0 or
                self.check_escape_route_recursive(
                    x + (0 if is_same_x else 1 if is_increase_x else -1),
                    y + (0 if is_same_y else 1 if is_increase_y else -1),
                    dist_from_bomb=dist_from_bomb + 1,
                    is_same_x=is_same_x,
                    is_same_y=is_same_y,
                    is_increase_x=is_increase_x,
                    is_increase_y=is_increase_y
                )
            )

    def check_if_put_bomb_have_escape(self):
        player_grid_pos_x = int(self.player.pos_x / Player.TILE_SIZE)
        player_grid_pos_y = int(self.player.pos_y / Player.TILE_SIZE)
        return (
            self.check_escape_route_recursive(
                player_grid_pos_x + 1, player_grid_pos_y, 0,
                is_same_x= False, is_same_y= True,
                is_increase_x= True, is_increase_y= False
            ) or self.check_escape_route_recursive(
                player_grid_pos_x - 1, player_grid_pos_y, 0,
                is_same_x= False, is_same_y= True,
                is_increase_x= False, is_increase_y= False
            ) or self.check_escape_route_recursive(
                player_grid_pos_x, player_grid_pos_y + 1, 0,
                is_same_x= True, is_same_y= False,
                is_increase_x= False, is_increase_y= True
            ) or self.check_escape_route_recursive(
                player_grid_pos_x, player_grid_pos_y - 1, 0,
                is_same_x= True, is_same_y= False,
                is_increase_x= False, is_increase_y= False
            )
        )

    # def getEnemyAtGridCoord(self, coord):
    #     for enemy in self.enemy_list:
    #         if (enemy.grid_x, enemy.grid_y) == coord:
    #             return enemy
    #     return None

    def target_enemy(self):
        enemy_grid_coords = [
            (enemy.get_grid_coords(), enemy)
            for enemy in self.enemy_list
        ]
        min_dist = np.inf
        target = None
        has_unobstructed_path = False

        # Always prefer an enemy to which you have
        # an unobstructed path
        for enemy_grid_coord, enemy in enemy_grid_coords:
            d = self.a_star_distance(
                self.player.get_grid_coords(), enemy_grid_coord
            )
            if d != np.inf:
                has_unobstructed_path = True
                if d < min_dist:
                    min_dist = d
                    target = enemy

        # Else, go by manhattan distance
        if not has_unobstructed_path:
            for enemy_grid_coord, enemy in enemy_grid_coords:
                d = manhattan_distance(
                    self.player.get_grid_coords(), enemy_grid_coord
                )
                if d < min_dist:
                    min_dist = d
                    target = enemy

        return target

    def step_in_direction(self, start_grid_coords, direction):
        """
        Simple utility function. Get coordinates of square
        in the direction of the start square.
        e.g. searchDirection([x, y], "left") --> [x-1, y]

        WARNING!!! If exceeding map dimensions, just returns
        the starting coordinates unchanged.
        """
        res = start_grid_coords
        max_x_dim = self.grid_state.shape[0]
        max_y_dim = self.grid_state.shape[1]
        match direction:
            case "left":
                res = (max(0, start_grid_coords[0] - 1), start_grid_coords[1])
            case "right":
                res = (min(max_x_dim, start_grid_coords[0] + 1), start_grid_coords[1])
            case "up":
                res = (start_grid_coords[0], max(0, start_grid_coords[1] - 1))
            case "down":
                res = (start_grid_coords[0], min(max_y_dim, start_grid_coords[1] + 1))
        return res

    def get_potential_bomb_sectors(self, coords, explosionRange):
        sectors = {coords}
        for direction in ["left", "right", "up", "down"]:
            pos = coords
            for _ in range(explosionRange):
                pos = self.step_in_direction(pos, direction)
                match self.grid_state[pos[0], pos[1]]:
                    case (
                        GridValues.EMPTY_GRID_VAL
                        | GridValues.BOX_GRID_VAL
                        | GridValues.BOMB_GRID_VAL
                        | GridValues.ENEMY_GRID_VAL
                        | GridValues.EXPLOSION_GRID_VAL
                    ):
                        sectors.add(pos)
                    case _:
                        break
        return sectors

    def get_periphery_of_sectors(self, sectors):
        allAdjacentSectors = set()
        for sector in sectors:
            # use a set to eliminate duplicates if stepInDirection() is next to map border
            adjacentSectors = {
                self.step_in_direction(sector, direction)
                for direction in ["left", "right", "up", "down"]
            }
            allAdjacentSectors.update(adjacentSectors)
        return allAdjacentSectors.difference(set(sectors))

    def get_filled_bounding_box_of_sectors(self, sectors):
        maxX = maxY = 0
        minX = minY = max(self.grid.shape[1], self.grid.shape[0])
        for sector in sectors:
            maxX = max(maxX, sector[0])
            maxY = max(maxY, sector[1])
            minX = min(minX, sector[0])
            minY = min(minY, sector[1])
        return {(x, y) for x in range(minX, maxX + 1) for y in range(minY, maxY + 1)}

    def consider_placing_bomb_now(self):
        stats_if_place_bomb_now = {
            "boxCount": 0,
            "enemyCount": 0,
            "enemyCountIfInfiniteRange": 0,
            "enemyCountMissByOne": 0,
            "enemyCountWithinSquare": 0,
        }

        box_coords = self.get_grid_coords_containing_value({GridValues.BOX_GRID_VAL})
        enemy_coords = [enemy.get_grid_coords() for enemy in self.enemy_list]
        potential_sectors = self.get_potential_bomb_sectors(
            self.player.get_grid_coords(), self.player.range
        )
        # print("boxes", boxCoords)
        # print("enemies", enemyCoords)

        # print("pot", potentialSectors)
        for coord in box_coords:
            if coord in potential_sectors:
                stats_if_place_bomb_now["boxCount"] += 1
        for coord in enemy_coords:
            if coord in potential_sectors:
                stats_if_place_bomb_now["enemyCount"] += 1

        potential_sectors_infinite_range = self.get_potential_bomb_sectors(
            self.player.get_grid_coords(), max(self.grid.shape[1], self.grid.shape[0])
        )
        # print("infinite", potentialSectorsInfiniteRange)
        for coord in enemy_coords:
            if coord in potential_sectors_infinite_range:
                stats_if_place_bomb_now["enemyCountIfInfiniteRange"] += 1

        potential_sectors_miss_by_one = self.get_periphery_of_sectors(potential_sectors)
        # print("missbyone", potentialSectorsMissByOne)
        for coord in enemy_coords:
            if coord in potential_sectors_miss_by_one:
                stats_if_place_bomb_now["enemyCountMissByOne"] += 1

        potential_sectors_within_square = self.get_filled_bounding_box_of_sectors(
            potential_sectors
        )
        # print("square", potentialSectorsWithinSquare)
        for coord in enemy_coords:
            if coord in potential_sectors_within_square:
                stats_if_place_bomb_now["enemyCountWithinSquare"] += 1

        return stats_if_place_bomb_now

    def get_grid_coords_containing_value(self, targetValues):
        width = self.grid.shape[0]
        height = self.grid.shape[1]
        res = []
        for x in range(width):
            for y in range(height):
                if self.grid_state[x,y] in targetValues:
                    res.append((x,y))
        return res

    def a_star_power_distance(self, entity1_mass, entity2_mass, entity1_grid_coords, entity2_grid_coords):
        gravity = 1.5
        a_star_path = self.a_star(entity1_grid_coords, entity2_grid_coords)
        if not a_star_path:
            return 0

        a_star_distance = len(a_star_path)
        return gravity ** a_star_distance

    def a_star_sigmoid(self, entity1_mass, entity2_mass, entity1_grid_coords, entity2GridCoords):
        gravity = 1
        a_star_path = self.a_star(entity1_grid_coords, entity2GridCoords)
        if not a_star_path:
            return 0

        a_star_distance = len(a_star_path)
        return (gravity * entity1_mass * entity2_mass) / (1 + np.e ** (a_star_distance + 10))

    def a_star_gravity(self, entity1_mass, entity2_mass, entity1_grid_coords, entity2GridCoords):
        gravity = 100
        a_star_path = self.a_star(entity1_grid_coords, entity2GridCoords)
        # print(aStarPath)
        if not a_star_path:
            return 0

        a_star_distance = len(a_star_path)
        # if aStarDistance == 0:
        #     aStarDistance = 0.1
        return (gravity * entity1_mass * entity2_mass) / (a_star_distance ** 2)

    @staticmethod
    def manhattan_power_distance(
        entity1_mass, entity2_mass, entity1_grid_coords, entity2_grid_coords
    ):
        gravity = 1.5
        manhattan = manhattan_distance(
            entity1_grid_coords, entity2_grid_coords
        )

        if manhattan == 0:
            manhattan = 0.1

        return gravity ** manhattan

    @staticmethod
    def manhattan_sigmoid(entity1_mass, entity2_mass, entity1_grid_coords, entity2_grid_coords):
        gravity = 100
        manhattan = manhattan_distance(entity1_grid_coords, entity2_grid_coords)
        if manhattan == 0:
            manhattan = 0.1
        return (gravity * entity1_mass * entity2_mass) / (1 + np.e ** (manhattan + 1))

    @staticmethod
    def manhattan_gravity(
        entity1_mass, entity2_mass, entity1_grid_coords,
        entity2_grid_coords
    ):
        gravity = 100
        manhattan = manhattan_distance(
            entity1_grid_coords, entity2_grid_coords
        )
        if manhattan == 0:
            manhattan = 0.1

        return (
            (gravity * entity1_mass * entity2_mass) /
            (manhattan ** 2)
        )

    def get_grid_coord_incentive_dict(self):
        res = {
            "box_gravity": 0,
            "enemy_gravity": 0,
            "target_enemy_gravity": 0,
            "bomb_gravity": 0,
        }

        coords = self.player.get_grid_coords()
        box_grid_coords = self.get_grid_coords_containing_value({
            GridValues.BOX_GRID_VAL
        })
        # print("boxes", boxGridCoords)
        for box in box_grid_coords:
            res["box_gravity"] += self.a_star_gravity(
                1, 1, box, coords
            )

        enemy_grid_coords = self.get_grid_coords_containing_value({
            GridValues.ENEMY_GRID_VAL
        })
        # print("enemies", enemyGridCoords)
        for enemy in enemy_grid_coords:
            res["enemy_gravity"] += self.a_star_sigmoid(
                1, 5, enemy, (coords[0], coords[1])
            )

        if self.currentTargetEnemy is not None:
            res ["target_enemy_gravity"] += self.manhattan_sigmoid(
                1,
                10,
                self.player.get_grid_coords(),
                self.currentTargetEnemy.get_grid_coords()
            )

        for bomb in self.bombs:
            bomb_coords = bomb.get_grid_coords()
            res["bomb_gravity"] += self.manhattan_power_distance(1, 10, bomb_coords, (coords[0], coords[1]))

        return res

    def get_grid_state_as_sectors(self):
        return {(x, y) for x in range(self.grid.shape[0]) for y in range(self.grid.shape[1])}

    def get_neighbours(self, coords):
        res = {
            self.step_in_direction(coords, direction)
            for direction in ["left", "right", "up", "down"]
        }
        return res

    def a_star(self, start, goal):
        free_sectors = self.get_grid_coords_containing_value(
            {GridValues.EMPTY_GRID_VAL, GridValues.PLAYER_GRID_VAL}
        )

        not_free_sectors = self.get_grid_state_as_sectors().difference(free_sectors)

        # if (start in notFreeSectors) or (goal in notFreeSectors):
        #     return None

        def get_neighbours_not_walls(coords):
            res = self.get_neighbours(coords).difference(not_free_sectors)
            return res

        # print(start, goal)
        # print(self.grid_state.T)
        path = find_path(
            start=start, goal=goal,
            neighbors_fnct=get_neighbours_not_walls,
            reversePath=False,
            heuristic_cost_estimate_fnct=manhattan_distance,
            distance_between_fnct=lambda a, b: 1,
            is_goal_reached_fnct=lambda a, b: a == b,
        )
        # print(list(path))
        if path is None:
            # print("NO PATH")
            return None
        else:
            # print(list(path))
            return list(path)

    def a_star_distance(self, start, goal):
        a_star_path = self.a_star(start, goal)
        return len(a_star_path) if a_star_path is not None else np.inf

    def square_is_walkable(self, coord):
        match self.grid_state[coord[0], coord[1]]:
            case GridValues.EMPTY_GRID_VAL | GridValues.PLAYER_GRID_VAL | 8:
                return True
            case _:
                return False

    def get_connected_walkable_squares(self, start_grid_coordinate):
        """
        Get all the walkable squares that are connected to an entity at startCoordinate.
        Uses flood fill algorithm; easy to overflow recursion but should be ok for our small map size
        """

        def flood_fill(s, coord):
            # no need to check if map dimensions exceeded, since stepInDirection
            # will just return the same coord, which should already be in s
            tup_coord = (coord[0], coord[1])  # for hashability to add to set

            walkable = False
            not_yet = False
            if self.square_is_walkable(coord):
                walkable = True
            # else:
            #     print("NOT WALKABLE:", coord)

            if tup_coord not in s:
                not_yet = True
            # else:
            #     print("ALREADY:", tupCoord)

            if walkable and not_yet:
                s.add(tup_coord)
                for direction in ["left", "right", "up", "down"]:
                    flood_fill(
                        s, self.step_in_direction(coord, direction)
                    )

        connected_walkable_squares = set()
        flood_fill(connected_walkable_squares, start_grid_coordinate)
        return connected_walkable_squares

    def get_leftover_walkable_squares(self):
        """
        Check if player has enough empty space to get away
        from the bombs that they set themselves.
        WARNING!!! Only checks for bombs that the  player sets for themselves.
        This is meant to help see if the player has put a bomb in such
        a way that there is no way for themselves to escape.
        """
        players_bombs = [bomb for bomb in self.bombs if bomb.bomber.is_player()]
        bomb_sectors = [
            (sectors[0], sectors[1])
            for bomb in players_bombs
            for sectors in bomb.sectors
        ]
        walkable_squares = self.get_connected_walkable_squares(self.player.get_grid_coords())
        leftover_walkable_squares = set(walkable_squares).difference(set(bomb_sectors))
        return leftover_walkable_squares

    def check_if_player_trapped_themselves(self):
        players_bombs = [bomb for bomb in self.bombs if bomb.bomber.is_player()]
        if len(players_bombs) == 0:
            return False

        return len(self.get_leftover_walkable_squares()) == 0

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
                    self.player_in_bomb_range = True
                    return True
        self.player_in_bomb_range = False
        return False

    def check_if_walking_to_bomb_range(self):
        player_pos_x = self.player.pos_x
        player_pos_y = self.player.pos_y

        if not self.player_in_bomb_range:
            for bomb in self.bombs:
                """
                bomb.sectors array stores all positions that the bomb
                explosion would hit.
                """
                for explosion_field_coords in bomb.sectors:
                    x, y = explosion_field_coords
                    in_bomb_range = (
                        int(player_pos_x / Player.TILE_SIZE) == x and
                        int(player_pos_y / Player.TILE_SIZE) == y
                    )

                    if in_bomb_range:
                        self.player_in_bomb_range = True
                        return True

        # If player is not walking into bomb range,
        # or is already in bomb range, return False
        self.player_in_bomb_range = False
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
        grid_x = self.player.grid_x
        grid_y = self.player.grid_y
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

    def check_if_waiting_beside_explosion(self, action):
        # This is specifically for Explosion class objects, not Bomb class objects
        grid_x = int(self.player.pos_x / Player.TILE_SIZE)
        grid_y = int(self.player.pos_y / Player.TILE_SIZE)
        top_pos = (grid_x, grid_y - 1)
        bottom_pos = (grid_x, grid_y + 1)
        left_pos = (grid_x - 1, grid_y)
        right_pos = (grid_x + 1, grid_y)

        if not self.player_in_bomb_range and action == self.WAIT:
            for explosion in self.explosions:
                """
                bomb.sectors array stores all positions that
                 the bomb explosion would hit.
                """
                for explosion_field_coords in explosion.sectors:
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

        # return (
        #     (top == GridValues.WALL_GRID_VAL or top == GridValues.BOX_GRID_VAL or top == GridValues.BOMB_GRID_VAL) and
        #     (bottom == GridValues.WALL_GRID_VAL or bottom == GridValues.BOX_GRID_VAL or bottom == GridValues.BOMB_GRID_VAL) and
        #     (left == GridValues.WALL_GRID_VAL or left == GridValues.BOX_GRID_VAL or left == GridValues.BOMB_GRID_VAL) and
        #     (right == GridValues.WALL_GRID_VAL or right == GridValues.BOX_GRID_VAL or right == GridValues.BOMB_GRID_VAL)
        # )

        obstacle_grid_values = [
            GridValues.WALL_GRID_VAL,
            GridValues.BOX_GRID_VAL,
            GridValues.BOMB_GRID_VAL
        ]

        return (
            top in obstacle_grid_values and
            bottom in obstacle_grid_values and
            left in obstacle_grid_values and
            right in obstacle_grid_values
        )

    def get_illegal_actions(self):
        illegal_actions = []

        if self.training_settings.IS_CHECKING_ILLEGAL_ACTION:
            player_pos_x = self.player.pos_x
            player_pos_y = self.player.pos_y

            grid_x = int(player_pos_x / Player.TILE_SIZE)
            grid_y = int(player_pos_y / Player.TILE_SIZE)
            top = self.grid[grid_x][grid_y-1]
            bottom = self.grid[grid_x][grid_y+1]
            left = self.grid[grid_x-1][grid_y]
            right = self.grid[grid_x+1][grid_y]

            obstacle_grid_values = [
                GridValues.WALL_GRID_VAL,
                GridValues.BOX_GRID_VAL,
                GridValues.BOMB_GRID_VAL,
                GridValues.ENEMY_GRID_VAL
            ]

            if top in obstacle_grid_values:
                illegal_actions.append(self.action_space_idx_map[self.UP])

            if bottom in obstacle_grid_values:
                illegal_actions.append(self.action_space_idx_map[self.DOWN])

            if left in obstacle_grid_values:
                illegal_actions.append(self.action_space_idx_map[self.LEFT])

            if right in obstacle_grid_values:
                illegal_actions.append(self.action_space_idx_map[self.RIGHT])

            if self.player.bomb_limit == 0:
                illegal_actions.append(self.action_space_idx_map[self.BOMB])

        return illegal_actions

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
                self.grid, self.bombs, self.explosions
            )

        if self.player.life:
            player_next_x = self.player_next_grid_pos_x
            player_next_y = self.player_next_grid_pos_y
            at_destination = (
                self.player_moving and
                int(self.player.pos_x / tile_size) == player_next_x and
                int(self.player.pos_y / tile_size) == player_next_y and
                self.player.pos_x % tile_size == 0 and
                self.player.pos_y % tile_size == 0
            )

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

        I: Incentives = self.incentives
        reward: float = 0

        if action == self.BOMB:
            can_plant_bomb = self.player.bomb_limit != 0 and self.player.life

            if can_plant_bomb:
                has_dropped_bomb = True
                self.player_in_bomb_range = True
                player_bomb = self.player.plant_bomb(self.grid)
                self.bombs.append(player_bomb)
                x = player_bomb.pos_x
                y = player_bomb.pos_y

                self.grid[x][y] = GridValues.BOMB_GRID_VAL
                # self.grid_state[x][y] = GridValues.BOMB_GRID_VAL
                self.player.bomb_limit -= 1

                place_bomb_rewards = self.consider_placing_bomb_now()
                # print(placeBomBNow)
                reward += place_bomb_rewards["boxCount"] * I.PLACE_BOMB_CONSIDER_BOX_COUNT
                reward += place_bomb_rewards["enemyCount"] * I.PLACE_BOMB_CONSIDER_ENEMY_COUNT
                reward += place_bomb_rewards["enemyCountIfInfiniteRange"] * I.PLACE_BOMB_CONSIDER_ENEMY_COUNT_INF_RANGE
                reward += place_bomb_rewards["enemyCountMissByOne"] * I.PLACE_BOMB_CONSIDER_ENEMY_COUNT_MISS_BY_ONE
                reward += place_bomb_rewards["enemyCountWithinSquare"] * I.PLACE_BOMB_CONSIDER_ENEMY_COUNT_IN_SQUARE

                if ((x, y) == (1, 1)) and (self.player_bombs_planted == 0):
                    reward += I.FIRST_CORNER_BOMB_PENALTY

                self.player_bombs_planted += 1

        update_bombs_result = self.update_bombs(dt)
        player_destroyed_boxes = update_bombs_result.player_destroyed_boxes
        player_kills = update_bombs_result.player_kills
        closeness = update_bombs_result.closeness_to_bomb
        self.player_boxes_destroyed += player_destroyed_boxes
        self.player_kills += update_bombs_result.player_kills

        bomb_closeness_reward = I.BOMB_DEATH_DISTANCE_PENALTY * closeness
        destroy_enemy_reward = I.DESTROY_ENEMY_REWARD * player_kills
        destroy_box_reward = I.DESTROY_BOX_REWARD * player_destroyed_boxes
        reward += destroy_enemy_reward
        reward += destroy_box_reward
        reward += bomb_closeness_reward

        """
        if bomb_closeness_reward != 0:
            print("CLOSENESS", bomb_closeness_reward)

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

        if not self.FLAG_checkIfPlayerTrappedThemselves:
            if self.check_if_player_trapped_themselves():
                self.FLAG_checkIfPlayerTrappedThemselves = True

        self.lifeDurationCounter += 1

        if self.gridCoordIncentiveTick == 0:
            gcid = self.get_grid_coord_incentive_dict()
            # print(gcid)
            reward += gcid["box_gravity"] * I.BOX_GRAVITY
            reward += gcid["enemy_gravity"] * I.ENEMY_GRAVITY
            reward += gcid["target_enemy_gravity"] * I.TARGET_ENEMY_GRAVITY
            reward += gcid["bomb_gravity"] * I.BOMB_GRAVITY
            self.gridCoordIncentiveTick = 1
        else:
            self.gridCoordIncentiveTick -= 1

        self.currentTargetEnemy = self.target_enemy()

        if action == self.BOMB and has_dropped_bomb:
            if self.check_if_put_bomb_have_escape():
                reward += I.PUT_BOMB_HAVE_ESCAPE_ROUTE_REWARD
            else:
                reward += I.PUT_BOMB_NO_ESCAPE_ROUTE_PENALTY

        if self.check_if_in_bomb_range():
            reward += I.IN_BOMB_RANGE_PENALTY
        else:
            reward += I.NOT_IN_BOMB_RANGE_REWARD

        if not self.player_in_bomb_range and self.check_if_walking_to_bomb_range():
            reward += I.MOVING_INTO_BOMB_RANGE_PENALTY
        elif self.player_in_bomb_range:
            if self.check_if_walking_out_of_bomb_range():
                reward += I.MOVING_FROM_BOMB_RANGE_REWARD
            else:
                reward += I.NOT_MOVING_FROM_BOMB_RANGE_PENALTY

        if not self.player_in_bomb_range and self.check_if_waiting_beside_bomb_range(action):
            reward += I.WAITING_BESIDE_BOMB_RANGE_REWARD

        if not self.player_in_bomb_range and self.check_if_waiting_beside_explosion(action):
            # This is specifically for Explosion class objects, not Bomb class objects
            reward += I.WAITING_BESIDE_EXPLOSION_REWARD

        if has_dropped_bomb and self.check_if_own_bomb_to_hit_boxes(player_bomb):
            reward += I.BOXES_IN_BOMB_RANGE_REWARD

        if self.check_if_walk_into_obj(action):
            reward += I.TRYING_TO_ENTER_WALL_PENALTY

        if self.check_if_trapped_with_bomb():
            reward += I.TRAPPED_WITH_BOMB_PENALTY

        if not self.player.life:
            reward += I.DEATH_PENALTY
            # print('DIE', I.DEATH_PENALTY)
            if self.FLAG_checkIfPlayerTrappedThemselves:
                reward += I.TRAPPED_THEMSELVES_GUARANTEED_DEATH
            reward += I.STAY_ALIVE * self.lifeDurationCounter
            self.clear_player_from_grid()

        self._score += reward
        # print(reward)
        return (
            self.get_normalised_state(), reward,
            self.is_game_ended(), self.player_moving
        )

    def get_normalised_state(self):
        raw_state = np.array(self.grid_state)
        one_hot_states = []
        """
        create one-hot encoded grids indicating which grid cells
        contain each type of object in the game
        """
        for grid_value in GridValues:
            grid_value = int(grid_value)
            one_hot = np.zeros_like(raw_state).astype(np.float32)
            one_hot[np.where(raw_state == grid_value)] = 1.0
            one_hot_states.append(one_hot)

        """
        store how long the bombs have been waiting in the grid
        (scaled from 0 to 1 (ready to explode)) 
        """
        all_bombs_grid = np.zeros_like(raw_state).astype(np.float32)
        player_bombs_grid = np.zeros_like(raw_state).astype(np.float32)
        bomb_waits = np.zeros_like(raw_state).astype(np.float32)

        for bomb in self.bombs:
            x, y = bomb.pos_x, bomb.pos_y
            all_bombs_grid[x][y] = 1.0
            player_bombs_grid[x][y] = 1.0 * int(bomb.is_player_bomb())
            bomb_waits[x][y] = (
                bomb.time_waited / Bomb.WAIT_DURATION
            )

        one_hot_states.append(all_bombs_grid)
        one_hot_states.append(player_bombs_grid)
        one_hot_states.append(bomb_waits)

        bomb_counts = np.zeros_like(raw_state).astype(np.float32)
        bomb_counts[:, :] = self.player.bomb_limit
        one_hot_states.append(bomb_counts)

        # stack the one-hot encoded grids into a 3D array
        xy_state = np.stack(one_hot_states, axis=0)
        """
        center the observation around the player
        to preserve spatial locality
        """
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
        self.player_bombs_planted = 0
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

        self.FLAG_checkIfPlayerTrappedThemselves = False
        self.lifeDurationCounter = 0

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
