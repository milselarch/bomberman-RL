import numpy as np
import pygame

from collections import deque
from matplotlib import pyplot as plt
from enums.algorithm import Algorithm
from game.BombermanEnv import BombermanEnv
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

    env = BombermanEnv(
        surface, show_path, player_alg, en1_alg,
        en2_alg, en3_alg, TILE_SIZE
    )

    # model hyperparameters
    ALPHA = 0.1
    EPS = 1.0

    Q = {}
    currentState = env.reset()
    tupleCurrentState = tuple(map(tuple, currentState))
    for action in env.action_space:
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
            action = env.max_action(Q, currentState, env.action_space) if rand < (1 - EPS) else env.actionSpaceSample()
            nextState, reward, done, info = env.step(action)
            epRewards += reward
            
            tupleCurrentState = tuple(map(tuple, currentState))
            tupleNextState = tuple(map(tuple, nextState))

            print((tupleCurrentState, action))

            actionTaken = env.max_action(Q, nextState, env.action_space)
            Q[tupleCurrentState, action] = Q[tupleCurrentState, action] + ALPHA*(reward + GAMMA * Q[tupleNextState, actionTaken] - Q[tupleCurrentState, action])
            currentState = nextState

        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0

        totalRewards[i] = epRewards

    plt.plot(totalRewards)
    plt.show()