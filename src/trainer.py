import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pygame

from collections import deque
from enums.algorithm import Algorithm
from BombermanEnv import BombermanEnv
from dqn import DQN


class Trainer(object):
    def __init__(self):
        self.learning_rate = 1e-4
        self.learning_rate_decay = 0.99
        self.exploration_decay = 0.95
        self.gamma = 0.975
        self.update_target_every = 10
        self.batch_size = 128
        self.EPISODES = 101

        pygame.display.init()
        self.pygame_info = pygame.display.Info()
        self.tile_size = int(self.pygame_info.current_h * 0.035)
        self.window_size = (13 * self.tile_size, 13 * self.tile_size)

        self.player_alg = Algorithm.PLAYER
        self.en1_alg = Algorithm.DIJKSTRA
        self.en2_alg = Algorithm.DFS
        self.en3_alg = Algorithm.DIJKSTRA
        self.show_path = True
        self.surface = pygame.display.set_mode(self.window_size)

        self.env = BombermanEnv(
            self.surface, self.show_path, self.player_alg,
            self.en1_alg, self.en2_alg, self.en3_alg,
            self.tile_size, tick_fps=0
        )

        self.agent = DQN(
            state_shape=self.env.stateShape,
            action_size=self.env.actionSpaceSize,
            batch_size=self.batch_size,
            learning_rate_max=self.learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            exploration_decay=self.exploration_decay,
            gamma=self.gamma
        )
        # agent.load(model_path)
        self.agent.save(f'models/-1.h5')

    def train(self):
        state = self.env.reset()
        state = np.expand_dims(state, axis=0)

        most_recent_losses = deque(maxlen=self.batch_size)

        log = []

        # fill up memory before training starts
        while self.agent.memory.length() < self.batch_size:
            action = self.agent.act(state)
            next_state, reward, done, game_info = self.env.step(
                self.env.actionSpace[action]
            )

            # Change state shape from (Height, Width) to (Height, Width, 1)
            next_state = np.expand_dims(next_state, axis=0)
            self.agent.remember(state, action, reward, next_state, done)
            state = next_state

        for e in range(0, self.EPISODES):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            step = 0
            ma_loss = None

            while not done:
                action = self.agent.act(state)
                step_result = self.env.step(self.env.actionSpace[action])
                next_state, reward, done, game_info = step_result
                next_state = np.expand_dims(next_state, axis=0)
                self.agent.remember(state, action, reward, next_state, done)

                state = next_state
                step += 1

                loss = self.agent.replay(episode=e)
                most_recent_losses.append(loss)
                ma_loss = np.array(most_recent_losses).mean()

                if loss is not None:
                    print(
                        f"Step: {step}. -- Loss: {loss}", end="          \r"
                    )

                if done:
                    print(
                        f"Episode {e}/{self.EPISODES - 1} completed "
                        f"with {step} steps. "
                        f"LR: {self.agent.learning_rate:.6f}. "
                        f"EP: {self.agent.exploration_rate:.2f}. "
                        f"MA loss: {ma_loss:.6f}"
                    )
                    break

            log.append([
                e, step, self.agent.learning_rate,
                self.agent.exploration_rate, ma_loss])

            self.agent.save(f'models/{e}.h5')