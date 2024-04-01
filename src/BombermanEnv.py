import random
import time
import numpy as np
import pygame
import matplotlib.pyplot as plt

from typing import List
from enums.algorithm import Algorithm
from player import Player
from enemy import Enemy
from explosion import Explosion
from bomb import Bomb

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
    def __init__(
        self, surface, path, player_alg, en1_alg, en2_alg,
        en3_alg, scale, tick_fps: int = 15
    ):
        self.tick_fps = tick_fps
        self.surface = surface
        self.path = path
        self.scale = scale

        # self.grid = [row[:] for row in GRID_BASE]
        self.grid = GRID_BASE.copy()
        self.generateMap()
        # self.grid = np.array(self.grid)
        self.gridState = self.grid.copy()
        self.reward = 0

        self.WALL_GRID_VAL = 1
        self.BOX_GRID_VAL = 2
        self.BOMB_GRID_VAL = 3
        self.EXPLOSION_GRID_VAL = 9
        self.ENEMY_GRID_VAL = 4
        self.PLAYER_GRID_VAL = 5
        # 1 is for indestructible walls
        # 2 is for destructible walls
        # 3 is for bombs
        # 4 for enemies
        # 5 for player
        # 4+3=7 for enemy dropping bomb
        # 5+3=8 for player dropping bomb
        # 9 for explosion. 
        # So max is 9.
        self.MAX_VAL_IN_GRID = self.EXPLOSION_GRID_VAL

        self.m = len(self.grid)  # Height of whole map, ie. no. of rows
        self.n = len(self.grid[0])  # Width of whole map, ie. no. of columns/fields in each row
        self.stateShape = (self.m, self.n, 1)
        self.UP = 'U'
        self.DOWN = 'D'
        self.LEFT = 'L'
        self.RIGHT = 'R'
        self.BOMB = 'Bomb'
        self.WAIT = 'Wait'

        self.actionSpace = [
            self.UP, self.DOWN, self.LEFT, self.RIGHT,
            self.BOMB, self.WAIT
        ]

        self.actionSpaceSize = len(self.actionSpace)
        self.actionsShape = (self.actionSpaceSize,)
        self.clock = pygame.time.Clock()
        self.gameEnded = False

        # self.font = pygame.font.SysFont('Bebas', scale)
        pygame.init()
        self.font = pygame.font.SysFont('Arial', scale)

        self.explosions: List[Explosion] = []
        self.bombs: List[Bomb] = []
        self.powerUps = []
        self.bombs.clear()
        self.explosions.clear()
        self.powerUps.clear()

        self.enemyList = []
        self.enemyBlocks = []
        self.enemiesPrevGridPosX = []
        self.enemiesPrevGridPosY = []

        self.player = Player()
        self.playerPrevGridPosX = int(self.player.pos_x / Player.TILE_SIZE)
        self.playerPrevGridPosY = int(self.player.pos_y / Player.TILE_SIZE)
        self.playerDirection_X = 0
        self.playerDirection_Y = 0
        self.playerMoving = False
        self.playerMovingAction = ''
        self.currentPlayerDirection = 0
        self.playerNextGridPos_X = None
        self.playerNextGridPos_Y = None
        self.playerInBombRange = False

        self.stepsPlayerInSamePos = 0
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
            self.enemyList.append(en1)
            self.enemyBlocks.append(en1)
            self.enemiesPrevGridPosX.append(int(en1.pos_x / Enemy.TILE_SIZE))
            self.enemiesPrevGridPosY.append(int(en1.pos_y / Enemy.TILE_SIZE))

        if self.en2_alg is not Algorithm.NONE:
            en2 = Enemy(1, 11, self.en2_alg)
            en2.load_animations('2', self.scale)
            self.enemyList.append(en2)
            self.enemyBlocks.append(en2)
            self.enemiesPrevGridPosX.append(int(en2.pos_x / Enemy.TILE_SIZE))
            self.enemiesPrevGridPosY.append(int(en2.pos_y / Enemy.TILE_SIZE))

        if self.en3_alg is not Algorithm.NONE:
            en3 = Enemy(11, 1, self.en3_alg)
            en3.load_animations('3', self.scale)
            self.enemyList.append(en3)
            self.enemyBlocks.append(en3)
            self.enemiesPrevGridPosX.append(int(en3.pos_x / Enemy.TILE_SIZE))
            self.enemiesPrevGridPosY.append(int(en3.pos_y / Enemy.TILE_SIZE))

        if self.player_alg is Algorithm.PLAYER:
            self.player.load_animations(self.scale)
            self.enemyBlocks.append(self.player)

        # elif self.player_alg is not Algorithm.NONE:
        #     en0 = Enemy(1, 1, self.player_alg)
        #     en0.load_animations('', self.scale)
        #     self.enemyList.append(en0)
        #     self.enemyBlocks.append(en0)
        #     self.player.life = False
        else:
            self.player.life = False

        self.setPlayerInGrid()
        self.setEnemiesInGrid()

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
        explosion1_img = pygame.transform.scale(explosion1_img, (scale, scale))

        explosion2_img = pygame.image.load('images/explosion/2.png')
        explosion2_img = pygame.transform.scale(explosion2_img, (scale, scale))

        explosion3_img = pygame.image.load('images/explosion/3.png')
        explosion3_img = pygame.transform.scale(explosion3_img, (scale, scale))

        self.terrainImages = [grass_img, block_img, box_img, grass_img]
        self.bombImages = [bomb1_img, bomb2_img, bomb3_img]
        self.explosionImages = [explosion1_img, explosion2_img, explosion3_img]

        power_up_bomb_img = pygame.image.load('images/power_up/bomb.png')
        power_up_bomb_img = pygame.transform.scale(power_up_bomb_img, (scale, scale))

        power_up_fire_img = pygame.image.load('images/power_up/fire.png')
        power_up_fire_img = pygame.transform.scale(power_up_fire_img, (scale, scale))

        self.powerUpsImages = [power_up_bomb_img, power_up_fire_img]

    def draw(self):
        ############################################
        ### FOR RENDERING THE GAME IN THE WINDOW ###
        ############################################

        BACKGROUND_COLOR = (107, 142, 35)

        self.surface.fill(BACKGROUND_COLOR)

        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                self.surface.blit(self.terrainImages[self.grid[i][j]],
                                  (i * self.scale, j * self.scale, self.scale, self.scale))

        for pu in self.powerUps:
            self.surface.blit(self.powerUpsImages[pu.type.value],
                              (pu.pos_x * self.scale, pu.pos_y * self.scale, self.scale, self.scale))

        for x in self.bombs:
            self.surface.blit(self.bombImages[x.frame],
                              (x.pos_x * self.scale, x.pos_y * self.scale, self.scale, self.scale))

        for y in self.explosions:
            for x in y.sectors:
                self.surface.blit(self.explosionImages[y.frame],
                                  (x[0] * self.scale, x[1] * self.scale, self.scale, self.scale))

        if self.player.life:
            self.surface.blit(self.player.animation[self.player.direction][self.player.frame],
                              (self.player.pos_x * (self.scale / 4), self.player.pos_y * (self.scale / 4), self.scale,
                               self.scale))

        for en in self.enemyList:
            if en.life:
                self.surface.blit(en.animation[en.direction][en.frame],
                                  (en.pos_x * (self.scale / 4), en.pos_y * (self.scale / 4), self.scale, self.scale))
                if self.path:
                    if en.algorithm == Algorithm.DFS:
                        for sek in en.path:
                            pygame.draw.rect(self.surface, (255, 0, 0, 240),
                                             [sek[0] * self.scale, sek[1] * self.scale, self.scale, self.scale], 1)
                    else:
                        for sek in en.path:
                            pygame.draw.rect(self.surface, (255, 0, 255, 240),
                                             [sek[0] * self.scale, sek[1] * self.scale, self.scale, self.scale], 1)

        if self.gameEnded:
            tf = self.font.render("Press ESC to go back to menu", False, (153, 153, 255))
            self.surface.blit(tf, (10, 10))

        pygame.display.update()

    def generateMap(self):
        ####################################################################
        """ This is just generating destroyable boxes if I am not wrong. """
        ####################################################################

        for i in range(1, len(self.grid) - 1):
            for j in range(1, len(self.grid[i]) - 1):
                if self.grid[i][j] != 0:
                    continue
                elif (i < 3 or i > len(self.grid) - 4) and (j < 3 or j > len(self.grid[i]) - 4):
                    continue
                if random.randint(0, 9) < 7:
                    self.grid[i][j] = 2

        return

    def setEnemiesInGrid(self):
        for i in range(len(self.enemyList)):
            self.gridState[self.enemiesPrevGridPosX[i]][self.enemiesPrevGridPosY[i]] -= self.ENEMY_GRID_VAL
            if self.gridState[self.enemiesPrevGridPosX[i]][self.enemiesPrevGridPosY[i]] < 0:
                self.gridState[self.enemiesPrevGridPosX[i]][self.enemiesPrevGridPosY[i]] = 0

            self.enemiesPrevGridPosX[i] = (int(self.enemyList[i].pos_x / Enemy.TILE_SIZE))
            self.enemiesPrevGridPosY[i] = (int(self.enemyList[i].pos_y / Enemy.TILE_SIZE))
            self.gridState[self.enemiesPrevGridPosX[i]][self.enemiesPrevGridPosY[i]] += self.ENEMY_GRID_VAL

    def clearEnemyFromGrid(self, enemy):
        enemyGridPosX = (int(enemy.pos_x / Enemy.TILE_SIZE))
        enemyGridPosY = (int(enemy.pos_y / Enemy.TILE_SIZE))

        if self.gridState[enemyGridPosX][enemyGridPosY] >= self.ENEMY_GRID_VAL:
            self.gridState[enemyGridPosX][enemyGridPosY] -= self.ENEMY_GRID_VAL

    def setPlayerInGrid(self):
        self.gridState[self.playerPrevGridPosX][self.playerPrevGridPosY] -= self.PLAYER_GRID_VAL
        if self.gridState[self.playerPrevGridPosX][self.playerPrevGridPosY] < 0:
            self.gridState[self.playerPrevGridPosX][self.playerPrevGridPosY] = 0

        self.playerPrevGridPosX = int(self.player.pos_x / Player.TILE_SIZE)
        self.playerPrevGridPosY = int(self.player.pos_y / Player.TILE_SIZE)
        self.gridState[self.playerPrevGridPosX][self.playerPrevGridPosY] += self.PLAYER_GRID_VAL

    def clearPlayerFromGrid(self):
        if self.gridState[self.playerPrevGridPosX][self.playerPrevGridPosY] >= self.PLAYER_GRID_VAL:
            self.gridState[self.playerPrevGridPosX][self.playerPrevGridPosX] -= self.PLAYER_GRID_VAL

    def setExplosionsInGrid(self):
        for i in range(len(self.explosions)):
            for gridCoordsTuple in self.explosions[i].sectors:
                self.gridState[gridCoordsTuple[0]][gridCoordsTuple[1]] = self.EXPLOSION_GRID_VAL

    def clearExplosionFromGrid(self, explosionObj):
        for gridCoordsTuple in explosionObj.sectors:
            # Set to 0 as nothing should be left if the explosion occurred on the grid square
            self.gridState[gridCoordsTuple[0]][gridCoordsTuple[1]] = 0

    def isGameEnded(self):
        if not self.player.life:
            return True

        for en in self.enemyList:
            if en.life:
                return False

        return True

    def updateBombs(self, dt):
        for bomb in self.bombs:
            bomb.update(dt)
            if bomb.time < 1:
                bomb.bomber.bomb_limit += 1
                self.grid[bomb.pos_x][bomb.pos_y] = 0
                self.gridState[bomb.pos_x][bomb.pos_y] -= self.BOMB_GRID_VAL
                if self.gridState[bomb.pos_x][bomb.pos_y] < 0:
                    self.gridState[bomb.pos_x][bomb.pos_y] = 0

                explosion = Explosion(bomb.pos_x, bomb.pos_y, bomb.range)
                explosion.explode(self.grid, self.bombs, bomb, self.powerUps)
                explosion.clear_sectors(self.grid, np.random, self.powerUps)
                self.explosions.append(explosion)

            elif bomb.time < 5:
                self.setExplosionsInGrid()

        if self.player not in self.enemyList:
            self.player.check_death(self.explosions)
            if not self.player.life:
                self.clearPlayerFromGrid()

        for enemy in self.enemyList:
            enemy.check_death(self.explosions)
            if not enemy.life:
                self.clearEnemyFromGrid(enemy)

        for explosion in self.explosions:
            explosion.update(dt)
            if explosion.time < 1:
                self.explosions.remove(explosion)
                self.clearExplosionFromGrid(explosion)

    def checkIfInBombRange(self):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
        for bomb in self.bombs:
            ######### bomb.sectors array stores all positions that the bomb explosion would hit. #########
            for explosionFieldCoords in bomb.sectors:
                explosionFieldPosX, explosionFieldPosY = explosionFieldCoords
                if int(playerPosX / Player.TILE_SIZE) == explosionFieldPosX and int(
                        playerPosY / Player.TILE_SIZE) == explosionFieldPosY:
                    return True
        return False

    def checkIfWalkingToBombRange(self):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y

        if not self.playerInBombRange:
            for bomb in self.bombs:
                ######### bomb.sectors array stores all positions that the bomb explosion would hit. #########
                for explosionFieldCoords in bomb.sectors:
                    explosionFieldPosX, explosionFieldPosY = explosionFieldCoords
                    if int(playerPosX / Player.TILE_SIZE) == explosionFieldPosX and int(
                            playerPosY / Player.TILE_SIZE) == explosionFieldPosY:
                        self.playerInBombRange = True
                        return True

        # If player is not walking into bomb range, or is already in bomb range, return False
        return False

    def checkIfWalkingOutOfBombRange(self):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y

        if self.playerInBombRange:
            for bomb in self.bombs:
                ######### bomb.sectors array stores all positions that the bomb explosion would hit. #########
                for explosionFieldCoords in bomb.sectors:
                    explosionFieldPosX, explosionFieldPosY = explosionFieldCoords
                    if int(playerPosX / Player.TILE_SIZE) == explosionFieldPosX and int(
                            playerPosY / Player.TILE_SIZE) == explosionFieldPosY:
                        # As long as player's grid is still in any explosion range, return false. 
                        return False
            # If player's grid is not in any explosion range and player was originally in bomb range, return true.
            self.playerInBombRange = False
            return True

        # If player is previously and currently not in bomb range, return False 
        return False

    def checkIfWaitingBesideBombRange(self, action):
        playerGridPosX = int(self.player.pos_x / Player.TILE_SIZE)
        playerGridPosY = int(self.player.pos_y / Player.TILE_SIZE)
        playerTopGridPos = (playerGridPosX, playerGridPosY - 1)
        playerBottomGridPos = (playerGridPosX, playerGridPosY + 1)
        playerLeftGridPos = (playerGridPosX - 1, playerGridPosY)
        playerRightGridPos = (playerGridPosX + 1, playerGridPosY)

        if not self.playerInBombRange and action == self.WAIT:
            for bomb in self.bombs:
                ######### bomb.sectors array stores all positions that the bomb explosion would hit. #########
                for explosionFieldCoords in bomb.sectors:
                    explosionFieldPosX, explosionFieldPosY = explosionFieldCoords
                    if (playerTopGridPos[0] == explosionFieldPosX and playerTopGridPos[1] == explosionFieldPosY) or \
                            (playerBottomGridPos[0] == explosionFieldPosX and playerBottomGridPos[
                                1] == explosionFieldPosY) or \
                            (playerLeftGridPos[0] == explosionFieldPosX and playerLeftGridPos[
                                1] == explosionFieldPosY) or \
                            (playerRightGridPos[0] == explosionFieldPosX and playerRightGridPos[
                                1] == explosionFieldPosY):
                        # If top, bottom, left or right grid of player's grid is in any explosion range, return True. 
                        return True
        return False

    def checkIfOwnBombToHitBoxes(self, playerBomb):
        # Only give reward when player just planted bomb
        playerGridPosX = int(self.player.pos_x / Player.TILE_SIZE)
        playerGridPosY = int(self.player.pos_y / Player.TILE_SIZE)

        if (playerGridPosX == playerBomb.pos_x and playerGridPosY == playerBomb.pos_y):
            for explosionFieldsCoords in playerBomb.sectors:
                explosionFieldsPosX, explosionFieldPosY = explosionFieldsCoords
                if self.grid[explosionFieldsPosX][explosionFieldPosY] == 2:
                    ###### 2 in grid means the field contains a destructable box. ###### 
                    return True

        return False

    def checkIfOwnBombToHitEnemy(self, playerBomb):
        for explosionFieldsCoords in playerBomb.sectors:
            explosionFieldsPosX, explosionFieldPosY = explosionFieldsCoords
            for enemy in self.enemyList:
                if enemy.pos_x == explosionFieldsPosX and enemy.pos_y == explosionFieldPosY:
                    return True

        return False

    def rewardIfOwnBombToHitEnemy(self, playerBomb):
        maxReward = 100
        minusPerFieldUnitDist = -20  # For every square in between bomb and enemy, subtract reward.
        outputReward = 0
        for explosionFieldsCoords in playerBomb.sectors:
            explosionFieldsPosX, explosionFieldPosY = explosionFieldsCoords
            for enemy in self.enemyList:
                if enemy.pos_x == explosionFieldsPosX and enemy.pos_y == explosionFieldPosY:
                    ################################################################################
                    """ Calculate number of squares between bomb and enemy
                        If bomb and enemy x position is not the same, calculate distance in x-axis 
                            as y positions would have to be the same for bomb to hit enemy.
                        Else calculate distance in y-axis."""
                    ################################################################################
                    fieldUnitDist = abs(playerBomb.pos_x - enemy.pos_x) if playerBomb.pos_x != enemy.pos_x else abs(
                        playerBomb.pos_y - enemy.pos_y)
                    outputReward += maxReward - minusPerFieldUnitDist * fieldUnitDist

        return outputReward

    def checkIfWalkIntoObj(self, action):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
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

        gridVal = self.grid[int(playerPosX / Player.TILE_SIZE) + x][int(playerPosY / Player.TILE_SIZE) + y]
        if gridVal == 1 or gridVal == 2 or gridVal == 3:
            return True
        else:
            return False

    def checkIfTrappedWithBomb(self):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
        top = self.grid[int(playerPosX / Player.TILE_SIZE)][int(playerPosY / Player.TILE_SIZE) - 1]
        bottom = self.grid[int(playerPosX / Player.TILE_SIZE)][int(playerPosY / Player.TILE_SIZE) + 1]
        left = self.grid[int(playerPosX / Player.TILE_SIZE) + 1][int(playerPosY / Player.TILE_SIZE)]
        right = self.grid[int(playerPosX / Player.TILE_SIZE) - 1][int(playerPosY / Player.TILE_SIZE)]

        if (top == 1 or top == 2 or top == 3) and \
                (bottom == 1 or bottom == 2 or bottom == 3) and \
                (left == 1 or left == 2 or left == 3) and \
                (right == 1 or right == 2 or right == 3):
            return True
        else:
            return False

    def checkIfWalkableSpace(self, action):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
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
        if self.grid[int(playerPosX / Player.TILE_SIZE) + x][int(playerPosY / Player.TILE_SIZE) + y] == 0:
            return True
        else:
            return False

    # def checkIfSamePlace(self, oldPosX, oldPosY):
    #     playerPosX = self.player.pos_x
    #     playerPosY = self.player.pos_y
    #     if int(playerPosX / Player.TILE_SIZE) == int(oldPosX / Player.TILE_SIZE) and int(playerPosY / Player.TILE_SIZE) == int(oldPosY / Player.TILE_SIZE):
    #         return True
    #     else:
    #         return False

    # def checkIfReachedDestSq(self, action):
    #     playerPosX = self.player.pos_x
    #     playerPosY = self.player.pos_y
    #     destSqPosX = self.destGridSqX
    #     destSqPosY = self.destGridSqY

    #     if int(playerPosX / Player.TILE_SIZE) == int(destSqPosX / Player.TILE_SIZE) and int(playerPosY / Player.TILE_SIZE) == int(destSqPosY / Player.TILE_SIZE):
    #         self.hasNoDestinationGrid = True

    #         # # If just reached in this action, then set destGrid
    #         # self.destGridSqX = self.player.pos_x
    #         # self.destGridSqY = self.player.pos_y
    #     else:
    #         self.hasNoDestinationGrid = False

    #     if self.hasNoDestinationGrid and (playerPosX != destSqPosX or playerPosY != destSqPosY):
    #         # If player is in the destination grid, but its exact position is not 
    #         x = 0
    #         y = 0
    #         if action == 'D':
    #             y = 1
    #         elif action == 'R':
    #             x = 1
    #         elif action == 'U':
    #             y = -1
    #         elif action == 'L':
    #             x = -1

    #         # If player is on destination grid square, but the exact position is not the center of the destination grid square, 
    #         # then set the destination grid square center pos X and Y as the grid square center pos X and Y that the player is moving towards now.
    #         gridVal = self.grid[int(destSqPosX / Player.TILE_SIZE) + (Player.TILE_SIZE) * x][int(destSqPosY / Player.TILE_SIZE) +  (Player.TILE_SIZE) * y]
    #         if gridVal != 1 and gridVal != 2 and gridVal != 3:
    #             self.destGridSqX = int(destSqPosX / Player.TILE_SIZE) + (Player.TILE_SIZE) * x
    #             self.destGridSqY = int(destSqPosY / Player.TILE_SIZE) +  (Player.TILE_SIZE) * y
    #             if action != 'Bomb' and action != 'Wait':
    #                 self.toDestGridAction = action
    #             else:
    #                 self.toDestGridAction = ''
    #         else:
    #             self.toDestGridAction = ''

    #         self.hasNoDestinationGrid = False

    #     return self.hasNoDestinationGrid

    def step(self, action):
        # print('TICK_FPS', self.tick_fps)
        dt = self.clock.tick(self.tick_fps)

        self.playerPrevPosX = self.player.pos_x
        self.playerPrevPosY = self.player.pos_y

        for enemy in self.enemyList:
            enemy.make_move(self.grid, self.bombs, self.explosions, self.enemyBlocks)

        if self.player.life:
            # currentPlayerDirection = self.player.direction
            # hasMovement = True
            # if action == 'D':
            #     currentPlayerDirection = 0
            #     self.player.move(0, 1, self.grid, self.enemyBlocks, self.powerUps)
            # elif action == 'R':
            #     currentPlayerDirection = 1
            #     self.player.move(1, 0, self.grid, self.enemyBlocks, self.powerUps)
            # elif action == 'U':
            #     currentPlayerDirection = 2
            #     self.player.move(0, -1, self.grid, self.enemyBlocks, self.powerUps)
            # elif action == 'L':
            #     currentPlayerDirection = 3
            #     self.player.move(-1, 0, self.grid, self.enemyBlocks, self.powerUps)
            # if action == 'Wait' or action == 'Bomb':
            #     hasMovement = False

            # if currentPlayerDirection != self.player.direction:
            #     self.player.frame = 0
            #     self.player.direction = currentPlayerDirection
            # if hasMovement:
            #     if self.player.frame == 2:
            #         self.player.frame = 0
            #     else:
            #         self.player.frame += 1

            if not self.playerMoving:
                # When player was originally not moving, or has reached its destination grid square, 
                # and an action to move is given as input, then set up values such that player would move in the same direction
                # until they reach the destination grid. 
                # THIS IS TO STOP THE PLAYER FROM STOPPING IN BETWEEN SQUARES.
                self.currentPlayerDirection = self.player.direction
                self.playerMoving = True

                self.playerDirection_X = 0
                self.playerDirection_Y = 0
                self.playerMovingAction = action

                if action == self.DOWN:
                    self.currentPlayerDirection = 0
                    self.playerDirection_X = 0
                    self.playerDirection_Y = 1
                elif action == self.RIGHT:
                    self.currentPlayerDirection = 1
                    self.playerDirection_X = 1
                    self.playerDirection_Y = 0
                elif action == self.UP:
                    self.currentPlayerDirection = 2
                    self.playerDirection_X = 0
                    self.playerDirection_Y = -1
                elif action == self.LEFT:
                    self.currentPlayerDirection = 3
                    self.playerDirection_X = -1
                    self.playerDirection_Y = 0
                elif action == self.WAIT or action == self.BOMB:
                    self.playerDirection_X = 0
                    self.playerDirection_Y = 0
                    self.playerMoving = False
                    self.playerMovingAction = ''

                if self.playerMoving:
                    # Storing Destination Grid Coordinates in Grid
                    self.playerNextGridPos_X = int(self.player.pos_x / Player.TILE_SIZE) + self.playerDirection_X
                    self.playerNextGridPos_Y = int(self.player.pos_y / Player.TILE_SIZE) + self.playerDirection_Y
                    gridVal = self.grid[self.playerNextGridPos_X][self.playerNextGridPos_Y]
                    if gridVal == 1 or gridVal == 2 or gridVal == 3:
                        # If Destination Grid is a Wall, Destructable Box
                        # or Bomb, Reset Values to not Force Player to Move
                        # in that Direction.
                        self.playerDirection_X = 0
                        self.playerDirection_Y = 0
                        self.playerNextGridPos_X = None
                        self.playerNextGridPos_Y = None
                        self.playerMoving = False
                        self.playerMovingAction = ''

            elif int(self.player.pos_x / Player.TILE_SIZE) == self.playerNextGridPos_X and \
                    int(self.player.pos_y / Player.TILE_SIZE) == self.playerNextGridPos_Y and \
                    self.player.pos_x % Player.TILE_SIZE == 0 and \
                    self.player.pos_y % Player.TILE_SIZE == 0:

                # If current grid coordinates of player is same as destination grid coordinates, 
                # and position of player are multiples of Player.TILE_SIZE,
                # THEN reset values 
                self.playerDirection_X = 0
                self.playerDirection_Y = 0
                self.playerNextGridPos_X = None
                self.playerNextGridPos_Y = None
                self.playerMoving = False
                self.playerMovingAction = ''
            else:
                action = self.playerMovingAction

            # Move player
            self.player.move(self.playerDirection_X, self.playerDirection_Y, self.grid, self.enemyBlocks, self.powerUps)

            if self.currentPlayerDirection != self.player.direction:
                self.player.frame = 0
                self.player.direction = self.currentPlayerDirection
            if self.playerMoving:
                if self.player.frame == 2:
                    self.player.frame = 0
                else:
                    self.player.frame += 1

            self.setEnemiesInGrid()
            self.setPlayerInGrid()

        ############################################
        """ FOR RENDERING THE GAME IN THE WINDOW """
        ############################################
        self.draw()

        hasDroppedBomb = False
        playerBomb = None

        if action == self.BOMB:
            if self.player.bomb_limit != 0 and self.player.life:
                hasDroppedBomb = True
                playerBomb = self.player.plant_bomb(self.grid)
                self.bombs.append(playerBomb)
                self.grid[playerBomb.pos_x][playerBomb.pos_y] = self.BOMB_GRID_VAL
                self.gridState[playerBomb.pos_x][playerBomb.pos_y] += self.BOMB_GRID_VAL
                self.player.bomb_limit -= 1

        self.updateBombs(dt)

        if not self.gameEnded:
            self.gameEnded = self.isGameEnded()

        ######################################
        """ REWARDS AND PENALTIES SECTION """
        ######################################
        IN_BOMB_RANGE_PENALTY = -5
        NOT_IN_BOMB_RANGE_PENALTY = 5
        MOVING_INTO_BOMB_RANGE_PENALTY = -10
        MOVING_FROM_BOMB_RANGE_REWARD = 10
        NOT_MOVING_FROM_BOMB_RANGE_PENALTY = -10
        WAITING_BESIDE_BOMB_RANGE_REWARD = 100
        TRYING_TO_ENTER_WALL_PENALTY = -5
        BOXES_IN_BOMB_RANGE_REWARD = 10
        WALK_INTO_SPACE_REWARD = 5
        SAME_GRID_PENALTY = -5
        TRAPPED_WITH_BOMB_PENALTY = -10
        DEATH_PENALTY = -100

        """ 
        Very high positive and negative rewards to prevent AI 
        from only moving a little in a direction before changing directions.
        """
        # NOT_MOVING_TO_DEST_GRID_PENALTY = -1000
        # MOVING_TO_DEST_GRID_PENALTY = 1000

        if not self.playerMoving:
            # Only give reward if player is not moving between grid squares.

            if self.checkIfInBombRange():
                self.reward += IN_BOMB_RANGE_PENALTY
            else:
                self.reward += NOT_IN_BOMB_RANGE_PENALTY

            if self.checkIfWalkingToBombRange():
                self.reward += MOVING_INTO_BOMB_RANGE_PENALTY

            if self.checkIfWalkingOutOfBombRange():
                self.reward += MOVING_FROM_BOMB_RANGE_REWARD
            else:
                self.reward += NOT_MOVING_FROM_BOMB_RANGE_PENALTY

            if self.checkIfWaitingBesideBombRange(action):
                self.reward += WAITING_BESIDE_BOMB_RANGE_REWARD

            if hasDroppedBomb and self.checkIfOwnBombToHitBoxes(playerBomb):
                self.reward += BOXES_IN_BOMB_RANGE_REWARD

            if hasDroppedBomb and self.checkIfOwnBombToHitEnemy(playerBomb):
                self.reward += self.rewardIfOwnBombToHitEnemy()

            if self.checkIfWalkIntoObj(action):
                self.reward += TRYING_TO_ENTER_WALL_PENALTY

            if self.checkIfTrappedWithBomb():
                self.reward += TRAPPED_WITH_BOMB_PENALTY

        # Need penalty codes for "if enemy very close / in an enclosure and action is not drop bomb" //////////////////////////////////////////////////////////

        if not self.player.life:
            self.reward += DEATH_PENALTY
            self.clearPlayerFromGrid()
        ######################################
        ######################################

        # print("NormalisedState: \n", np.array_str(self.getNormalisedState()))
        # print()
        # print("Player pos x, pos y: ", self.player.pos_x, self.player.pos_y)
        # print("Player int(self.player.pos_x / Player.TILE_SIZE), int(self.player.pos_y / Player.TILE_SIZE): ", int(self.player.pos_x / Player.TILE_SIZE), int(self.player.pos_y / Player.TILE_SIZE))
        # print("self.playerDirection_X, self.playerDirection_Y: ", self.playerDirection_X, self.playerDirection_Y)
        # print("self.playerNextGridPos_X, self.playerNextGridPos_Y:", self.playerNextGridPos_X, self.playerNextGridPos_Y)
        # print("self.playerMoving: ", self.playerMoving)
        # print("self.playerMovingAction: ", self.playerMovingAction)
        # print("action: ", action)

        # print("self.hasNoDestinationGrid: ", self.hasNoDestinationGrid)
        # print("self.toDestGridActio: ", self.toDestGridAction)
        # print("self.destGridSqX, self.destGridSqY: ", self.destGridSqX, ", ", self.destGridSqY)
        # print("Player int(self.destGridSqX / Player.TILE_SIZE), int(self.destGridSqY / Player.TILE_SIZE): ", int(self.destGridSqX / Player.TILE_SIZE), int(self.destGridSqY / Player.TILE_SIZE))
        # print("Grid x: ", len(self.grid[0]))
        # print("Grid y: ", len(self.grid))

        # print("self.reward: ", self.reward)
        # print()

        # if self.player.life:
        #     time.sleep(2)

        return self.getNormalisedState(), self.reward, self.isGameEnded(), self.playerMoving

    def getNormalisedState(self):
        return self.gridState  # / self.MAX_VAL_IN_GRID

    def reset(self):
        # self.grid = [row[:] for row in GRID_BASE]
        # self.grid = np.array(GRID_BASE)
        self.grid = GRID_BASE.copy()
        self.generateMap()
        self.gridState = self.grid.copy()
        self.reward = 0

        self.explosions.clear()
        self.bombs.clear()
        self.powerUps.clear()

        self.enemyList.clear()
        self.enemyBlocks.clear()
        self.enemiesPrevGridPosX.clear()
        self.enemiesPrevGridPosY.clear()

        self.player = Player()
        self.playerPrevGridPosX = int(self.player.pos_x / Player.TILE_SIZE)
        self.playerPrevGridPosY = int(self.player.pos_y / Player.TILE_SIZE)
        self.playerDirection_X = 0
        self.playerDirection_Y = 0
        self.playerMoving = False
        self.currentPlayerDirection = 0
        self.playerNextGridPos_X = None
        self.playerNextGridPos_Y = None

        # self.destGridSqX = self.player.pos_x
        # self.destGridSqY = self.player.pos_y
        # self.hasNoDestinationGrid = True
        # self.toDestGridAction = ''

        self.clock = pygame.time.Clock()
        self.gameEnded = False

        if self.en1_alg is not Algorithm.NONE:
            en1 = Enemy(11, 11, self.en1_alg)
            en1.load_animations('1', self.scale)
            self.enemyList.append(en1)
            self.enemyBlocks.append(en1)
            self.enemiesPrevGridPosX.append(int(en1.pos_x / Enemy.TILE_SIZE))
            self.enemiesPrevGridPosY.append(int(en1.pos_y / Enemy.TILE_SIZE))

        if self.en2_alg is not Algorithm.NONE:
            en2 = Enemy(1, 11, self.en2_alg)
            en2.load_animations('2', self.scale)
            self.enemyList.append(en2)
            self.enemyBlocks.append(en2)
            self.enemiesPrevGridPosX.append(int(en2.pos_x / Enemy.TILE_SIZE))
            self.enemiesPrevGridPosY.append(int(en2.pos_y / Enemy.TILE_SIZE))

        if self.en3_alg is not Algorithm.NONE:
            en3 = Enemy(11, 1, self.en3_alg)
            en3.load_animations('3', self.scale)
            self.enemyList.append(en3)
            self.enemyBlocks.append(en3)
            self.enemiesPrevGridPosX.append(int(en3.pos_x / Enemy.TILE_SIZE))
            self.enemiesPrevGridPosY.append(int(en3.pos_y / Enemy.TILE_SIZE))

        if self.player_alg is Algorithm.PLAYER:
            self.player.load_animations(self.scale)
            self.enemyBlocks.append(self.player)

        # elif self.player_alg is not Algorithm.NONE:
        #     en0 = Enemy(1, 1, self.player_alg)
        #     en0.load_animations('', self.scale)
        #     self.enemyList.append(en0)
        #     self.enemyBlocks.append(en0)
        #     self.player.life = False
        else:
            self.player.life = False

        self.setPlayerInGrid()
        self.setEnemiesInGrid()

        return self.getNormalisedState()

    def is_player_alive(self):
        return self.player.life

    def count_player_kills(self) -> int:
        player_kills = 0

        for enemy in self.enemyList:
            if enemy.killed_by_player:
                player_kills += 1

        return player_kills

    def actionSpaceSample(self):
        #####################################
        """ Just randomly take any action """
        #####################################
        return np.random.choice(self.actionSpace)

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
