import numpy as np
import pygame
import dqn

from enums.algorithm import Algorithm
from dqn import DQN
from world_for_deep_q_learning import BombermanEnv

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

model_path = "models/196.h5"

env = BombermanEnv(surface, show_path, player_alg, en1_alg, en2_alg, en3_alg, TILE_SIZE)
agent = DQN(
    state_shape=env.stateShape,
    action_size=env.actionSpaceSize
)
agent.load(model_path)

state = env.reset()
state = np.expand_dims(state, axis=0)
done = False

pygame.init()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if done:
        state = env.reset()
        state = np.expand_dims(state, axis=0)
    action = agent.act(state)
    state, reward, done, gameinfo = env.step(env.actionSpace[action])
    state = np.expand_dims(state, axis=0)
    pygame.display.flip()

pygame.quit()