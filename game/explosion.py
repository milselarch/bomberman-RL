from typing import Union, Optional, List

from game.GridValues import GridValues
from enums.power_up_type import PowerUpType
from game.power_up import PowerUp
from game.bomb import Bomb


class Explosion:
    bomber = None

    def __init__(self, x: int, y: int, r: int):
        self.source_x = x
        self.source_y = y
        self.range = r
        self.time = 300
        self.frame = 0
        self.sectors = []

    def explode(self, map, bombs: List[Bomb], b: Bomb, power_ups):
        self.bomber = b.bomber
        self.sectors.extend(b.sectors)
        bombs.remove(b)
        self.bomb_chain(bombs, map, power_ups)

    def bomb_chain(self, bombs: List[Bomb], map, power_ups):
        for s in self.sectors:
            for x in power_ups:
                if x.pos_x == s[0] and x.pos_y == s[1]:
                    power_ups.remove(x)

            for x in bombs:
                if x.pos_x == s[0] and x.pos_y == s[1]:
                    map[x.pos_x][x.pos_y] = 0
                    x.bomber.bomb_limit += 1
                    self.explode(map, bombs, x, power_ups)

    def clear_sectors(self, map, random, power_ups) -> int:
        destroyed_boxes = 0

        for i in self.sectors:
            if map[i[0]][i[1]] == GridValues.BOX_GRID_VAL:
                destroyed_boxes += 1

                r = random.randint(0, 9)
                if r == 0:
                    power_ups.append(PowerUp(i[0], i[1], PowerUpType.BOMB))
                elif r == 1:
                    power_ups.append(PowerUp(i[0], i[1], PowerUpType.FIRE))

            map[i[0]][i[1]] = GridValues.EMPTY_GRID_VAL

        return destroyed_boxes

    def update(self, dt):
        self.time = self.time - dt

        if self.time < 100:
            self.frame = 2
        elif self.time < 200:
            self.frame = 1