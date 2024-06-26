from typing import List, Tuple

from game.Actor import Actor


class Bomb:
    WAIT_DURATION = 3000
    frame = 0

    def __init__(self, r: int, x: int, y: int, grid_map, bomber: Actor):
        self.range = r
        self.pos_x = x
        self.pos_y = y
        self.time = self.WAIT_DURATION
        self.bomber: Actor = bomber
        self.sectors = []
        self.get_range(grid_map)

    def is_player_bomb(self):
        return self.bomber.is_player()

    @property
    def time_waited(self):
        # time waited since being planted
        return self.WAIT_DURATION - self.time

    def update(self, dt):
        self.time = self.time - dt

        if self.time < 1000:
            self.frame = 2
        elif self.time < 2000:
            self.frame = 1

    def get_range(self, map):
        self.sectors.append([self.pos_x, self.pos_y])
        # print(self.pos_x, self.pos_y)

        for x in range(1, self.range):
            if map[self.pos_x + x][self.pos_y] == 1:
                break
            elif map[self.pos_x + x][self.pos_y] == 0 or map[self.pos_x - x][self.pos_y] == 3:
                self.sectors.append([self.pos_x + x, self.pos_y])
            elif map[self.pos_x + x][self.pos_y] == 2:
                self.sectors.append([self.pos_x + x, self.pos_y])
                break
        for x in range(1, self.range):
            if map[self.pos_x - x][self.pos_y] == 1:
                break
            elif map[self.pos_x - x][self.pos_y] == 0 or map[self.pos_x - x][self.pos_y] == 3:
                self.sectors.append([self.pos_x - x, self.pos_y])
            elif map[self.pos_x - x][self.pos_y] == 2:
                self.sectors.append([self.pos_x - x, self.pos_y])
                break
        for x in range(1, self.range):
            if map[self.pos_x][self.pos_y + x] == 1:
                break
            elif map[self.pos_x][self.pos_y + x] == 0 or map[self.pos_x][self.pos_y + x] == 3:
                self.sectors.append([self.pos_x, self.pos_y + x])
            elif map[self.pos_x][self.pos_y + x] == 2:
                self.sectors.append([self.pos_x, self.pos_y + x])
                break
        for x in range(1, self.range):
            if map[self.pos_x][self.pos_y - x] == 1:
                break
            elif map[self.pos_x][self.pos_y - x] == 0 or map[self.pos_x][self.pos_y - x] == 3:
                self.sectors.append([self.pos_x, self.pos_y - x])
            elif map[self.pos_x][self.pos_y - x] == 2:
                self.sectors.append([self.pos_x, self.pos_y - x])
                break

    @property
    def grid_x(self) -> int:
        return int(self.pos_x / 4)

    @property
    def grid_y(self) -> int:
        return int(self.pos_y / 4)

    def getGridCoords(self):
        return (self.grid_x, self.grid_y)
