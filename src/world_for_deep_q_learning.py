import random
import numpy as np
import pygame
import matplotlib.pyplot as plt
from enums.algorithm import Algorithm
from player import Player
from enemy import Enemy
from explosion import Explosion

GRID_BASE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

class BombermanEnv(object):
    def __init__(self, surface, path, 
                    player_alg, en1_alg, en2_alg, 
                    en3_alg, scale):

        self.surface = surface
        self.path = path
        self.scale = scale

        self.grid = [row[:] for row in GRID_BASE]
        print("BEFORE: ", self.grid)
        self.generateMap()
        print("AFTER: ", self.grid)

        self.grid = np.array(self.grid)
        
        self.MAX_VAL_IN_GRID = 3    # 1 is for indestructable walls, 2 is for destructable walls, 3 is for bombs, so 3 is the max value I know 
        self.m = len(self.grid)     # Height of whole map, ie. no. of rows
        self.n = len(self.grid[0])  # Width of whole map, ie. no. of columns/fields in each row
        self.stateShape = (self.m, self.n, 1)
        self.actionSpace = ['U', 'D', 'L', 'R', 'Bomb', 'Wait']
        self.actionSpaceSize = len(self.actionSpace)
        self.actionsShape = (self.actionSpaceSize, )

        self.clock = pygame.time.Clock()
        self.gameEnded = False

        # self.font = pygame.font.SysFont('Bebas', scale)
        pygame.init()
        self.font = pygame.font.SysFont('Arial', scale)
        

        self.enemyList = []
        self.player = Player()
        self.enemyBlocks = []

        self.explosions = []
        self.bombs = []
        self.powerUps = []
        self.bombs.clear()
        self.explosions.clear()
        self.powerUps.clear()

        self.player = Player()
        self.playerPrevPosX = self.player.pos_x
        self.playerPrevPosY = self.player.pos_y
        self.stepsPlayerInSamePos = 0
        self.destGridSqX = self.player.pos_x
        self.destGridSqY = self.player.pos_y
        self.hasNoDestinationGrid = True
        self.toDestGridAction = ''

        self.en1_alg = en1_alg
        self.en2_alg = en2_alg
        self.en3_alg = en3_alg
        self.player_alg = player_alg

        if self.en1_alg is not Algorithm.NONE:
            en1 = Enemy(11, 11, self.en1_alg)
            en1.load_animations('1', self.scale)
            self.enemyList.append(en1)
            self.enemyBlocks.append(en1)

        if self.en2_alg is not Algorithm.NONE:
            en2 = Enemy(1, 11, self.en2_alg )
            en2.load_animations('2', self.scale)
            self.enemyList.append(en2)
            self.enemyBlocks.append(en2)

        if self.en3_alg is not Algorithm.NONE:
            en3 = Enemy(11, 1, self.en3_alg)
            en3.load_animations('3', self.scale)
            self.enemyList.append(en3)
            self.enemyBlocks.append(en3)

        if self.player_alg is Algorithm.PLAYER:
            self.player.load_animations(self.scale)
            self.enemyBlocks.append(self.player)
        elif self.player_alg is not Algorithm.NONE:
            en0 = Enemy(1, 1, self.player_alg)
            en0.load_animations('', self.scale)
            self.enemyList.append(en0)
            self.enemyBlocks.append(en0)
            self.player.life = False
        else:
            self.player.life = False

        
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
                self.surface.blit(self.terrainImages[self.grid[i][j]], (i * self.scale, j * self.scale, self.scale, self.scale))

        for pu in self.powerUps:
            self.surface.blit(self.powerUpsImages[pu.type.value], (pu.pos_x * self.scale, pu.pos_y * self.scale, self.scale, self.scale))

        for x in self.bombs:
            self.surface.blit(self.bombImages[x.frame], (x.pos_x * self.scale, x.pos_y * self.scale, self.scale, self.scale))

        for y in self.explosions:
            for x in y.sectors:
                self.surface.blit(self.explosionImages[y.frame], (x[0] * self.scale, x[1] * self.scale, self.scale, self.scale))
        if self.player.life:
            self.surface.blit(self.player.animation[self.player.direction][self.player.frame],
                (self.player.pos_x * (self.scale / 4), self.player.pos_y * (self.scale / 4), self.scale, self.scale))
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
                explosion = Explosion(bomb.pos_x, bomb.pos_y, bomb.range)
                explosion.explode(self.grid, self.bombs, bomb, self.powerUps)
                explosion.clear_sectors(self.grid, np.random, self.powerUps)
                self.explosions.append(explosion)
        if self.player not in self.enemyList:
            self.player.check_death(self.explosions)
        for enemy in self.enemyList:
            enemy.check_death(self.explosions)
        for explosion in self.explosions:
            explosion.update(dt)
            if explosion.time < 1:
                self.explosions.remove(explosion)

    def checkIfInBombRange(self):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
        for bomb in self.bombs:
            ######### bomb.sectors array stores all positions that the bomb explosion would hit. #########
            for explosionFieldCoords in bomb.sectors:
                explosionFieldPosX, explosionFieldPosY = explosionFieldCoords
                if int(playerPosX / Player.TILE_SIZE) == explosionFieldPosX and int(playerPosY / Player.TILE_SIZE) == explosionFieldPosY:
                    return True
        return False 
    
    def checkIfWalkingToBombRange(self, action):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
        x = 0
        y = 0
        if action == 'D':
            y = 1
        elif action == 'R':
            x = 1
        elif action == 'U':
            y = -1
        elif action == 'L':
            x = -1

        for bomb in self.bombs:
            ######### bomb.sectors array stores all positions that the bomb explosion would hit. #########
            for explosionFieldCoords in bomb.sectors:
                explosionFieldPosX, explosionFieldPosY = explosionFieldCoords
                if int(playerPosX / Player.TILE_SIZE) + x == explosionFieldPosX and int(playerPosY / Player.TILE_SIZE) + y == explosionFieldPosY:
                    return True
        return False 
    
    def checkIfWalkingOutOfBombRange(self, action):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
        x = 0
        y = 0
        if action == 'D':
            y = 1
        elif action == 'R':
            x = 1
        elif action == 'U':
            y = -1
        elif action == 'L':
            x = -1

        if self.checkIfInBombRange():
            for bomb in self.bombs:
                ######### bomb.sectors array stores all positions that the bomb explosion would hit. #########
                for explosionFieldCoords in bomb.sectors:
                    explosionFieldPosX, explosionFieldPosY = explosionFieldCoords
                    if self.checkIfWalkableSpace(action) and int(playerPosX / Player.TILE_SIZE) + x == explosionFieldPosX and int(playerPosY / Player.TILE_SIZE) + y == explosionFieldPosY:
                        # As long as player's next destination grid is still in any explosion range, return false. 
                        return False
            # If player's next destination grid is not in any explosion range and player can move into that grid, return true.
            return self.checkIfWalkableSpace(action) 
    
    def checkIfOwnBombToHitBoxes(self, playerBomb):
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
                    fieldUnitDist = abs(playerBomb.pos_x - enemy.pos_x) if playerBomb.pos_x != enemy.pos_x else abs(playerBomb.pos_y - enemy.pos_y)
                    outputReward += maxReward - minusPerFieldUnitDist * fieldUnitDist
        return outputReward    
    
    def checkIfWalkIntoObj(self, action):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
        x = 0
        y = 0
        if action == 'D':
            y = 1
        elif action == 'R':
            x = 1
        elif action == 'U':
            y = -1
        elif action == 'L':
            x = -1

        gridVal = self.grid[int(playerPosX / Player.TILE_SIZE) + x][int(playerPosY / Player.TILE_SIZE) + y] 
        if gridVal == 1 or gridVal == 2 or gridVal == 3:
            return True
        else:
            return False
    
    def checkIfWalkableSpace(self, action):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
        x = 0
        y = 0
        if action == 'D':
            y = 1
        elif action == 'R':
            x = 1
        elif action == 'U':
            y = -1
        elif action == 'L':
            x = -1
        if self.grid[int(playerPosX / Player.TILE_SIZE) + x][int(playerPosY / Player.TILE_SIZE) + y] == 0:
            return True
        else:
            return False
        
    def checkIfSamePlace(self, oldPosX, oldPosY):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
        if int(playerPosX / Player.TILE_SIZE) == int(oldPosX / Player.TILE_SIZE) and int(playerPosY / Player.TILE_SIZE) == int(oldPosY / Player.TILE_SIZE):
            return True
        else:
            return False
        
    def checkIfReachedDestSq(self, action):
        playerPosX = self.player.pos_x
        playerPosY = self.player.pos_y
        destSqPosX = self.destGridSqX
        destSqPosY = self.destGridSqY

        if int(playerPosX / Player.TILE_SIZE) == int(destSqPosX / Player.TILE_SIZE) and int(playerPosY / Player.TILE_SIZE) == int(destSqPosY / Player.TILE_SIZE):
            self.hasNoDestinationGrid = True
            
            # # If just reached in this action, then set destGrid
            # self.destGridSqX = self.player.pos_x
            # self.destGridSqY = self.player.pos_y
        else:
            self.hasNoDestinationGrid = False

        if self.hasNoDestinationGrid and (playerPosX != destSqPosX or playerPosY != destSqPosY):
            # If player is in the destination grid, but its exact position is not 
            x = 0
            y = 0
            if action == 'D':
                y = 1
            elif action == 'R':
                x = 1
            elif action == 'U':
                y = -1
            elif action == 'L':
                x = -1
            
            # If player is on destination grid square, but the exact position is not the center of the destination grid square, 
            # then set the destination grid square center pos X and Y as the grid square center pos X and Y that the player is moving towards now.
            gridVal = self.grid[int(destSqPosX / Player.TILE_SIZE) + (Player.TILE_SIZE) * x][int(destSqPosY / Player.TILE_SIZE) +  (Player.TILE_SIZE) * y]
            if gridVal != 1 and gridVal != 2 and gridVal != 3:
                self.destGridSqX = int(destSqPosX / Player.TILE_SIZE) + (Player.TILE_SIZE) * x
                self.destGridSqY = int(destSqPosY / Player.TILE_SIZE) +  (Player.TILE_SIZE) * y
                if action != 'Bomb' and action != 'Wait':
                    self.toDestGridAction = action
                else:
                    self.toDestGridAction = ''
            else:
                self.toDestGridAction = ''

            self.hasNoDestinationGrid = False


        return self.hasNoDestinationGrid

    def step(self, action):
        dt = self.clock.tick(15)

        self.playerPrevPosX = self.player.pos_x
        self.playerPrevPosY = self.player.pos_y

        for enemy in self.enemyList:
            enemy.make_move(self.grid, self.bombs, self.explosions, self.enemyBlocks)

        if self.player.life:
            currentPlayerDirection = self.player.direction
            hasMovement = False

            if action == 'D':
                currentPlayerDirection = 0
                self.player.move(0, 1, self.grid, self.enemyBlocks, self.powerUps)
                hasMovement = True
            elif action == 'R':
                currentPlayerDirection = 1
                self.player.move(1, 0, self.grid, self.enemyBlocks, self.powerUps)
                hasMovement = True
            elif action == 'U':
                currentPlayerDirection = 2
                self.player.move(0, -1, self.grid, self.enemyBlocks, self.powerUps)
                hasMovement = True
            elif action == 'L':
                currentPlayerDirection = 3
                self.player.move(-1, 0, self.grid, self.enemyBlocks, self.powerUps)
                hasMovement = True
            elif action == 'Wait':
                hasMovement = False
            if currentPlayerDirection != self.player.direction:
                self.player.frame = 0
                self.player.direction = currentPlayerDirection
            if hasMovement:
                if self.player.frame == 2:
                    self.player.frame = 0
                else:
                    self.player.frame += 1

        ############################################
        """ FOR RENDERING THE GAME IN THE WINDOW """
        ############################################
        self.draw()
        

        hasDroppedBomb = False
        playerBomb = None
        if action == 'Bomb':
            if self.player.bomb_limit != 0 and self.player.life:
                hasDroppedBomb = True
                playerBomb = self.player.plant_bomb(self.grid)
                self.bombs.append(playerBomb)
                self.grid[playerBomb.pos_x][playerBomb.pos_y] = 3
                self.player.bomb_limit -= 1

        self.updateBombs(dt)

        if not self.gameEnded:
            self.gameEnded = self.isGameEnded()

        ######################################
        """ REWARDS AND PENALTIES SECTION """
        ######################################
        reward = 0
        IN_BOMB_RANGE_PENALTY = -50
        MOVING_INTO_BOMB_RANGE_PENALTY = -50
        MOVING_FROM_BOMB_RANGE_REWARD = 1000
        NOT_MOVING_FROM_BOMB_RANGE_PENALTY = -10
        TRYING_TO_ENTER_WALL_PENALTY = -50
        BOXES_IN_BOMB_RANGE_REWARD = 10
        WALK_INTO_SPACE_REWARD = 50
        SAME_GRID_PENALTY = -5
        DEATH_PENALTY = -1000

        # Very high positive and negative rewards to prevent AI from only moving a little in a direction before changing directions.
        NOT_MOVING_TO_DEST_GRID_PENALTY = -1000
        MOVING_TO_DEST_GRID_PENALTY = 1000
        # if not self.checkIfReachedDestSq(action):
        #     if not self.checkIfWalkingToBombRange(action) and action != self.toDestGridAction:
        #         reward += NOT_MOVING_TO_DEST_GRID_PENALTY
        #     else:
        #         reward += MOVING_TO_DEST_GRID_PENALTY

        if self.checkIfInBombRange():
            reward += IN_BOMB_RANGE_PENALTY

        if self.checkIfWalkingToBombRange(action):
            reward += MOVING_INTO_BOMB_RANGE_PENALTY

        if self.checkIfWalkingOutOfBombRange(action):
            reward += MOVING_FROM_BOMB_RANGE_REWARD
        else:
            reward += NOT_MOVING_FROM_BOMB_RANGE_PENALTY

        if hasDroppedBomb and self.checkIfOwnBombToHitBoxes(playerBomb):
            reward += BOXES_IN_BOMB_RANGE_REWARD

        if hasDroppedBomb and self.checkIfOwnBombToHitEnemy(playerBomb):
            reward += self.rewardIfOwnBombToHitEnemy()

        if self.checkIfWalkIntoObj(action):
            reward += TRYING_TO_ENTER_WALL_PENALTY

        if self.checkIfWalkableSpace(action):
            if self.checkIfSamePlace(self.playerPrevPosX, self.playerPrevPosY):
                # self.stepsPlayerInSamePos += 10
                reward += SAME_GRID_PENALTY  ######## - self.stepsPlayerInSamePos
            else:
                self.stepsPlayerInSamePos = 0
                reward += WALK_INTO_SPACE_REWARD

        # if enemy very close / in an enclosure and action is not drop bomb PENALTY //////////////////////////////////////////////////////////

        if not self.player.life:
            reward += DEATH_PENALTY
        ######################################
            
        
        # print("NormalisedState: ", np.array_str(self.getNormalisedState()))
        print()
        print("Player pos x, pos y: ", self.player.pos_x, self.player.pos_y)
        print("Player int(self.player.pos_x / Player.TILE_SIZE), int(self.player.pos_y / Player.TILE_SIZE): ", int(self.player.pos_x / Player.TILE_SIZE), int(self.player.pos_y / Player.TILE_SIZE))
        # print("self.hasNoDestinationGrid: ", self.hasNoDestinationGrid)
        # print("self.toDestGridActio: ", self.toDestGridAction)
        # print("self.destGridSqX, self.destGridSqY: ", self.destGridSqX, ", ", self.destGridSqY)
        # print("Player int(self.destGridSqX / Player.TILE_SIZE), int(self.destGridSqY / Player.TILE_SIZE): ", int(self.destGridSqX / Player.TILE_SIZE), int(self.destGridSqY / Player.TILE_SIZE))
        # print("Grid x: ", len(self.grid[0]))
        # print("Grid y: ", len(self.grid))
        print("Reward: ", reward)
        

        return self.getNormalisedState(), reward, self.isGameEnded(), None

    def getNormalisedState(self):
        return self.grid # / self.MAX_VAL_IN_GRID

    def reset(self):
        self.grid = [row[:] for row in GRID_BASE]
        self.generateMap()
        self.grid = np.array(self.grid)

        self.enemyList.clear()
        self.enemyBlocks.clear()
        self.explosions.clear()
        self.bombs.clear()
        self.powerUps.clear()

        self.player = Player()
        self.playerPrevPosX = self.player.pos_x
        self.playerPrevPosY = self.player.pos_y
        self.stepsPlayerInSamePos = 0

        self.destGridSqX = self.player.pos_x
        self.destGridSqY = self.player.pos_y
        self.hasNoDestinationGrid = True
        self.toDestGridAction = ''

        self.clock = pygame.time.Clock()
        self.gameEnded = False

        if self.en1_alg is not Algorithm.NONE:
            en1 = Enemy(11, 11, self.en1_alg)
            en1.load_animations('1', self.scale)
            self.enemyList.append(en1)
            self.enemyBlocks.append(en1)

        if self.en2_alg is not Algorithm.NONE:
            en2 = Enemy(1, 11, self.en2_alg )
            en2.load_animations('2', self.scale)
            self.enemyList.append(en2)
            self.enemyBlocks.append(en2)

        if self.en3_alg is not Algorithm.NONE:
            en3 = Enemy(11, 1, self.en3_alg)
            en3.load_animations('3', self.scale)
            self.enemyList.append(en3)
            self.enemyBlocks.append(en3)

        if self.player_alg is Algorithm.PLAYER:
            self.player.load_animations(self.scale)
            self.enemyBlocks.append(self.player)
        elif self.player_alg is not Algorithm.NONE:
            en0 = Enemy(1, 1, self.player_alg)
            en0.load_animations('', self.scale)
            self.enemyList.append(en0)
            self.enemyBlocks.append(en0)
            self.player.life = False
        else:
            self.player.life = False

        return self.getNormalisedState()

    def actionSpaceSample(self):
        #####################################
        """ Just randomly take any action """
        #####################################
        return np.random.choice(self.actionSpace)

    def maxAction(self, Q, state, actions):
        #####################################################################################################################################
        """ Make an array of values for all actions at a particular state, 
            with each value calculated with:
            Q[state, action] + ALPHA*(reward + GAMMA * Q[state, actionOfPrevMaxValue] - Q[state, action]) """
        #####################################################################################################################################
        
        state = tuple(map(tuple, state))

        for a in actions:
            if (state, a) not in Q:
                Q[state, a] = 0
        
        values = np.array([Q[state, a] for a in actions]) 
        action = np.argmax(values)
        return actions[action]
