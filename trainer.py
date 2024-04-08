import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pygame
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from game.Incentives import Incentives
from collections import deque
from datetime import datetime as Datetime
from enums.algorithm import Algorithm
from game.BombermanEnv import BombermanEnv
from memory_profiler import profile as profile_memory
from dqn import DQN

fp = open('memory_profiler.log', 'w')


class Trainer(object):
    def __init__(
        self, name='ddqn', incentives: Incentives = Incentives()
    ):
        self.name = name
        self.incentives = incentives

        self.learning_rate = 1e-4
        self.learning_rate_decay = 1  # 0.99
        self.exploration_decay = 0.999  # 0.95
        self.exploration_max = 0.2
        self.exploration_min = 0.001  # 0.01
        self.gamma = 0.995  # 0.975
        self.update_target_every = 10
        self.episode_buffer_size = 128
        self.episodes = 20 * 1000

        self.logs_dir = 'logs'
        self.models_dir = 'models'
        self.date_stamp = self.make_date_stamp()

        self.log_dir = None
        self.model_dir = None
        self.t_logs_writer = None
        self.v_logs_writer = None
        self.init_tensorboard()

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
            self.tile_size, incentives=incentives,
            simulate_time=True, physics_fps=15,
            render_fps=15
        )

        self.agent = DQN(
            state_shape=self.env.state_shape,
            action_size=self.env.action_space_size,
            batch_size=self.episode_buffer_size,
            learning_rate_max=self.learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            exploration_decay=self.exploration_decay,
            exploration_min=self.exploration_min,
            exploration_max=self.exploration_max,
            gamma=self.gamma
        )
        # agent.load(model_path)
        self.agent.save(f'saves/-1.h5')

    @staticmethod
    def make_date_stamp():
        return Datetime.now().strftime("%y%m%d-%H%M")

    @staticmethod
    def kwargify(**kwargs):
        return kwargs

    def init_tensorboard(self):
        dir_save_name = f'{self.name}-{self.date_stamp}'
        self.log_dir = f'logs/{dir_save_name}'
        self.model_dir = f'models/{dir_save_name}'

        train_path = self.log_dir + '/training'
        valid_path = self.log_dir + '/validation'
        self.t_logs_writer = tf.summary.create_file_writer(train_path)
        self.v_logs_writer = tf.summary.create_file_writer(valid_path)

    @profile_memory(stream=fp)
    def train(self):
        state = self.env.reset()
        state = np.expand_dims(state, axis=0)
        most_recent_losses = deque(maxlen=self.episode_buffer_size)
        best_score = -float('inf')

        # fill up memory before training starts
        while self.agent.memory.length() < self.episode_buffer_size:
            action = self.agent.act(state)
            next_state, reward, done, game_info = self.env.step(
                self.env.action_space[action]
            )

            # Change state shape from (Height, Width) to (Height, Width, 1)
            next_state = np.expand_dims(next_state, axis=0)
            self.agent.remember(state, action, reward, next_state, done)
            state = next_state

        pbar = tqdm(range(self.episodes))

        for e in pbar:
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            ma_loss = None
            done = False
            step = 0
            loss = 0

            while not done:
                action = self.agent.act(state)
                step_result = self.env.step(self.env.action_space[action])
                next_state, reward, done, game_info = step_result
                next_state = np.expand_dims(next_state, axis=0)
                self.agent.remember(state, action, reward, next_state, done)

                state = next_state
                step += 1

                most_recent_losses.append(loss)
                ma_loss = np.array(most_recent_losses).mean()

                if loss is not None:
                    pbar.set_description(
                        f"[{self.date_stamp}] "
                        f"Step: {step}. -- Loss: {loss:.6f}"
                    )

            loss = self.agent.replay(episode_no=e)
            game_score = self.env.get_score()

            episode_kills = self.env.count_player_kills()
            kill_score = episode_kills
            is_alive = self.env.is_player_alive()
            live_tag = 'alive' if is_alive else 'dead'
            if not is_alive:
                kill_score -= 1

            self.write_logs(
                file_writer=self.t_logs_writer, episode_no=e,
                loss_value=loss, kill_score=kill_score, is_alive=is_alive,
                game_duration=self.env.steps, kills=self.env.player_kills,
                boxes_destroyed=self.env.player_boxes_destroyed,
                score=game_score
            )

            print(
                f"Episode {e}/{self.episodes - 1} completed "
                f"with {step} steps. "
                f"LR: {self.agent.learning_rate:.3f}. "
                f"EP: {self.agent.exploration_rate:.2f}. "
                f"Kills: {episode_kills} [{live_tag}] "
                f"boxes: {self.env.player_boxes_destroyed} "
                f"score: {game_score:.3f} "
            )

            if game_score > best_score:
                best_score = game_score
                loss_tag = f'{best_score:.4f}'.replace('.', '_')
                save_path = f'{self.model_dir}/{e}-L{loss_tag}.h5'
                print('new best model saved to:', save_path)
                self.agent.save(save_path)

    @staticmethod
    def write_logs(
        file_writer, episode_no: int,
        loss_value: float, kill_score: float, is_alive: int,
        game_duration: int, kills: int, boxes_destroyed: int,
        score: float
    ):
        with file_writer.as_default():
            tf.summary.scalar(
                'loss', data=loss_value, step=episode_no
            )
            tf.summary.scalar(
                'kill_score', data=kill_score, step=episode_no
            )
            tf.summary.scalar(
                'is_alive', data=is_alive, step=episode_no
            )
            tf.summary.scalar(
                'game_duration', data=game_duration, step=episode_no
            )
            tf.summary.scalar(
                'kills', data=kills, step=episode_no
            )
            tf.summary.scalar(
                'boxes_destroyed', data=boxes_destroyed, step=episode_no
            )
            tf.summary.scalar(
                'score', data=score, step=episode_no
            )