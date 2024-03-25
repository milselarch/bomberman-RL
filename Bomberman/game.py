from typing import Optional

import pygame
import sys
import random

from gym import Env, spaces
from pygame import Surface

from enums.power_up_type import PowerUpType
from player import Player
from explosion import Explosion
from enemy import Enemy
from enums.algorithm import Algorithm
from power_up import PowerUp

BACKGROUND_COLOR = (107, 142, 35)
GRID_BASE = [
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


"""
font = None
player = None

enemy_list = []
ene_blocks = []
bombs = []
explosions = []
power_ups = []
"""


class BaseGame(Env):
    def __init__(
        self, player_alg, en1_alg, en2_alg, en3_alg, scale
    ):
        super().__init__()
        self.player_alg = player_alg
        self.en1_alg = en1_alg
        self.en2_alg = en2_alg
        self.en3_alg = en3_alg
        self.scale = scale

        self.action_space = spaces.Discrete(start=0, n=5)
        self.player = None
        self.en0 = None
        self.en1 = None
        self.en2 = None
        self.en3 = None

        self.enemy_list = []
        self.ene_blocks = []
        self.bombs = []
        self.explosions = []
        self.power_ups = []
        self.grid = []

    def reset(
        self, seed: Optional[int] = None,
        options: Optional[int] = None
    ):
        self.enemy_list = []
        self.ene_blocks = []
        self.bombs.clear()
        self.explosions.clear()
        self.power_ups.clear()
        self.player = Player()

        if self.en1_alg is not Algorithm.NONE:
            self.en1 = Enemy(11, 11, self.en1_alg)
            # self.en1.load_animations('1', self.scale)
            self.enemy_list.append(self.en1)
            self.ene_blocks.append(self.en1)

        if self.en2_alg is not Algorithm.NONE:
            self.en2 = Enemy(1, 11, self.en2_alg)
            # self.en2.load_animations('2', self.scale)
            self.enemy_list.append(self.en2)
            self.ene_blocks.append(self.en2)

        if self.en3_alg is not Algorithm.NONE:
            self.en3 = Enemy(11, 1, self.en3_alg)
            # self.en3.load_animations('3', self.scale)
            self.enemy_list.append(self.en3)
            self.ene_blocks.append(self.en3)

        if self.player_alg is Algorithm.PLAYER:
            # self.player.load_animations(self.scale)
            self.ene_blocks.append(self.player)
        elif self.player_alg is not Algorithm.NONE:
            self.en0 = Enemy(1, 1, self.player_alg)
            # self.en0.load_animations('', self.scale)
            self.enemy_list.append(self.en0)
            self.ene_blocks.append(self.en0)
            self.player.life = False
        else:
            self.player.life = False

        self.grid = [row[:] for row in GRID_BASE]
        generate_map(self.grid)
        self.player.reset()

    def update_bombs(self, grid, dt):
        for b in self.bombs:
            b.update(dt)
            if b.time < 1:
                b.bomber.bomb_limit += 1
                grid[b.pos_x][b.pos_y] = 0
                exp_temp = Explosion(b.pos_x, b.pos_y, b.range)
                exp_temp.explode(grid, self.bombs, b, self.power_ups)
                exp_temp.clear_sectors(grid, random, self.power_ups)
                self.explosions.append(exp_temp)

        if self.player not in self.enemy_list:
            self.player.check_death(self.explosions)
        for en in self.enemy_list:
            en.check_death(self.explosions)
        for e in self.explosions:
            e.update(dt)
            if e.time < 1:
                self.explosions.remove(e)

    def check_end_game(self):
        if not self.player.life:
            return True

        for en in self.enemy_list:
            if en.life:
                return False

        return True

    def done(self):
        return self.check_end_game()


class Game(BaseGame):
    def __init__(
        self, *args, surface: Surface, scale: int,
        show_path=True, **kwargs
    ):
        super(Game, self).__init__(*args, **kwargs, scale=scale)
        self.surface = surface
        self.show_path = show_path
        self.scale = scale
        self.font = None

        self.grass_img = pygame.image.load('images/terrain/grass.png')
        self.grass_img = pygame.transform.scale(
            self.grass_img, (self.scale, self.scale)
        )

        self.block_img = pygame.image.load('images/terrain/block.png')
        self.block_img = pygame.transform.scale(
            self.block_img, (self.scale, self.scale)
        )

        self.box_img = pygame.image.load('images/terrain/box.png')
        self.box_img = pygame.transform.scale(
            self.box_img, (self.scale, self.scale)
        )

        self.bomb1_img = pygame.image.load('images/bomb/1.png')
        self.bomb1_img = pygame.transform.scale(
            self.bomb1_img, (self.scale, self.scale)
        )

        self.bomb2_img = pygame.image.load('images/bomb/2.png')
        self.bomb2_img = pygame.transform.scale(
            self.bomb2_img, (self.scale, self.scale)
        )

        self.bomb3_img = pygame.image.load('images/bomb/3.png')
        self.bomb3_img = pygame.transform.scale(
            self.bomb3_img, (self.scale, self.scale)
        )

        self.explosion1_img = pygame.image.load('images/explosion/1.png')
        self.explosion1_img = pygame.transform.scale(
            self.explosion1_img, (self.scale, self.scale)
        )

        self.explosion2_img = pygame.image.load('images/explosion/2.png')
        self.explosion2_img = pygame.transform.scale(
            self.explosion2_img, (self.scale, self.scale)
        )

        self.explosion3_img = pygame.image.load('images/explosion/3.png')
        self.explosion3_img = pygame.transform.scale(
            self.explosion3_img, (self.scale, self.scale)
        )

        self.terrain_images = [
            self.grass_img, self.block_img, self.box_img, self.grass_img
        ]
        self.bomb_images = [
            self.bomb1_img, self.bomb2_img, self.bomb3_img
        ]
        self.explosion_images = [
            self.explosion1_img, self.explosion2_img, self.explosion3_img
        ]

        self.power_up_bomb_img = pygame.image.load('images/power_up/bomb.png')
        self.power_up_bomb_img = pygame.transform.scale(
            self.power_up_bomb_img, (self.scale, self.scale)
        )

        self.power_up_fire_img = pygame.image.load('images/power_up/fire.png')
        self.power_up_fire_img = pygame.transform.scale(
            self.power_up_fire_img, (self.scale, self.scale)
        )

        self.power_ups_images = [
            self.power_up_bomb_img, self.power_up_fire_img
        ]

        self.reset()

    def reset(
        self, seed: Optional[int] = None,
        options: Optional[int] = None
    ):
        self.font = pygame.font.SysFont('Bebas', self.scale)
        super().reset()

        if self.en1 is not None:
            self.en1.load_animations('1', self.scale)
        if self.en2 is not None:
            self.en2.load_animations('2', self.scale)
        if self.en3 is not None:
            self.en3.load_animations('3', self.scale)

        if self.player_alg is Algorithm.PLAYER:
            self.player.load_animations(self.scale)
        if self.en0 is not None:
            self.en0.load_animations('', self.scale)

    def step(self, action_no):
        raise NotImplementedError

    def draw(
        self, s, grid, tile_size, show_path, game_ended, terrain_images,
        bomb_images, explosion_images, power_ups_images
    ):
        player = self.player

        s.fill(BACKGROUND_COLOR)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                s.blit(
                    terrain_images[grid[i][j]],
                    (i * tile_size, j * tile_size, tile_size, tile_size)
                )

        for pu in self.power_ups:
            s.blit(power_ups_images[pu.type.value], (
                pu.pos_x * tile_size, pu.pos_y * tile_size,
                tile_size, tile_size
            ))

        for x in self.bombs:
            s.blit(bomb_images[x.frame], (
                x.pos_x * tile_size, x.pos_y * tile_size,
                tile_size, tile_size
            ))

        for y in self.explosions:
            for x in y.sectors:
                s.blit(explosion_images[y.frame], (
                    x[0] * tile_size, x[1] * tile_size,
                    tile_size, tile_size
                ))

        if player.life:
            s.blit(player.animation[player.direction][player.frame], (
                player.pos_x * (tile_size / 4),
                player.pos_y * (tile_size / 4), tile_size, tile_size
            ))

        for en in self.enemy_list:
            if not en.life:
                continue

            s.blit(
                en.animation[en.direction][en.frame], (
                    en.pos_x * (tile_size / 4),
                    en.pos_y * (tile_size / 4),
                    tile_size, tile_size
                )
            )

            if not show_path:
                continue

            if en.algorithm == Algorithm.DFS:
                for sek in en.path:
                    pygame.draw.rect(
                        s, (255, 0, 0, 240),
                        [sek[0] * tile_size, sek[1] * tile_size,
                         tile_size, tile_size], 1
                    )
            else:
                for sek in en.path:
                    pygame.draw.rect(
                        s, (255, 0, 255, 240),
                        [sek[0] * tile_size, sek[1] * tile_size,
                         tile_size, tile_size], 1
                    )

        if game_ended:
            tf = self.font.render(
                "Press ESC to go back to menu", False, (153, 153, 255)
            )
            s.blit(tf, (10, 10))

        pygame.display.update()

    def start(self):
        self.main(
            s=self.surface, tile_size=self.scale,
            show_path=self.show_path,
            terrain_images=self.terrain_images,
            bomb_images=self.bomb_images,
            explosion_images=self.explosion_images,
            power_ups_images=self.power_ups_images
        )

    def main(
        self, s: Surface, tile_size: int, show_path: bool,
        terrain_images, bomb_images,
        explosion_images, power_ups_images
    ):
        self.reset()
        # power_ups.append(PowerUp(1, 2, PowerUpType.BOMB))
        # power_ups.append(PowerUp(2, 1, PowerUpType.FIRE))
        clock = pygame.time.Clock()

        running = True
        game_ended = False

        while running:
            dt = clock.tick(15)
            for en in self.enemy_list:
                en.make_move(
                    self.grid, self.bombs, self.explosions,
                    self.ene_blocks
                )

            if self.player.life:
                keys = pygame.key.get_pressed()
                temp = self.player.direction
                movement = False

                if keys[pygame.K_DOWN]:
                    temp = 0
                    movement = True
                    self.player.move(
                        0, 1, self.grid, self.ene_blocks, self.power_ups
                    )
                elif keys[pygame.K_RIGHT]:
                    temp = 1
                    movement = True
                    self.player.move(
                        1, 0, self.grid, self.ene_blocks, self.power_ups
                    )
                elif keys[pygame.K_UP]:
                    temp = 2
                    movement = True
                    self.player.move(
                        0, -1, self.grid, self.ene_blocks, self.power_ups
                    )
                elif keys[pygame.K_LEFT]:
                    temp = 3
                    movement = True
                    self.player.move(
                        -1, 0, self.grid, self.ene_blocks, self.power_ups
                    )

                if temp != self.player.direction:
                    self.player.frame = 0
                    self.player.direction = temp
                if movement:
                    if self.player.frame == 2:
                        self.player.frame = 0
                    else:
                        self.player.frame += 1

            self.draw(
                s, self.grid, tile_size, show_path, game_ended,
                terrain_images, bomb_images, explosion_images,
                power_ups_images
            )

            if not game_ended:
                game_ended = self.check_end_game()

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE:
                        cannot_plant_bomb = (
                            self.player.bomb_limit == 0 or
                            not self.player.life
                        )

                        if cannot_plant_bomb:
                            continue

                        temp_bomb = self.player.plant_bomb(self.grid)
                        self.bombs.append(temp_bomb)
                        self.grid[temp_bomb.pos_x][temp_bomb.pos_y] = 3
                        self.player.bomb_limit -= 1
                    elif e.key == pygame.K_ESCAPE:
                        running = False

            self.update_bombs(self.grid, dt)

        # self.explosions.clear()
        # self.enemy_list.clear()
        # self.ene_blocks.clear()
        # self.power_ups.clear()


def generate_map(grid):
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[i]) - 1):
            if grid[i][j] != 0:
                continue

            is_out_of_bounds = (
                (i < 3 or i > len(grid) - 4) and
                (j < 3 or j > len(grid[i]) - 4)
            )

            if is_out_of_bounds:
                continue
            if random.randint(0, 9) < 7:
                grid[i][j] = 2




