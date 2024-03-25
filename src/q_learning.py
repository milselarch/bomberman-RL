import numpy as np
from collections import deque
from matplotlib import pyplot as plt
import pygame


from enums.algorithm import Algorithm
from world_for_deep_q_learning import BombermanEnv
from dqn import DQN

if __name__ == '__main__':
    LEARNING_RATE = 1e-4
    LEARNING_RATE_DECAY = 0.99
    EXPLORATION_DECAY = 0.95
    GAMMA = 0.975
    UPDATE_TARGET_EVERY = 10

    BATCH_SIZE = 128
    EPISODES = 101

    pygame.display.init()
    INFO = pygame.display.Info()
    TILE_SIZE = int(INFO.current_h * 0.035)
    WINDOW_SIZE = (13 * TILE_SIZE, 13 * TILE_SIZE)
    player_alg = Algorithm.PLAYER
    en1_alg = Algorithm.DIJKSTRA
    en2_alg = Algorithm.DFS
    en3_alg = Algorithm.DIJKSTRA
    show_path = True
    surface = pygame.display.set_mode(WINDOW_SIZE)

    env = BombermanEnv(surface, show_path, player_alg, en1_alg, en2_alg, en3_alg, TILE_SIZE)


    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0

    Q = {}
    currentState = env.reset()
    tupleCurrentState = tuple(map(tuple, currentState))
    for action in env.actionSpace:
        Q[tupleCurrentState, action] = 0

    numGames = 50000
    totalRewards = np.zeros(numGames)
    for i in range(numGames):
        if i % 5000 == 0:
            print('starting game ', i)
        done = False
        epRewards = 0
        currentState = env.reset()
        while not done:
            rand = np.random.random()
            action = env.maxAction(Q, currentState, env.actionSpace) if rand < (1-EPS) else env.actionSpaceSample()
            nextState, reward, done, info = env.step(action)
            epRewards += reward
            
            tupleCurrentState = tuple(map(tuple, currentState))
            tupleNextState = tuple(map(tuple, nextState))

            actionTaken = env.maxAction(Q, nextState, env.actionSpace)
            Q[tupleCurrentState, action] = Q[tupleCurrentState, action] + ALPHA*(reward + GAMMA * Q[tupleNextState, actionTaken] - Q[tupleCurrentState, action])
            currentState = nextState
        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards

    plt.plot(totalRewards)
    plt.show()