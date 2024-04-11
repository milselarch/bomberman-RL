import os

from Transition import Transition

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


class Trainer(object):
    def __init__(
        self, name='ddqn', incentives: Incentives = Incentives()
    ):
        self.name = name
        self.incentives = incentives

        self.learning_rate = 0.01
        self.exploration_decay = 0.9995  # 0.95
        self.exploration_max = 0.2
        self.exploration_min = 0.001  # 0.01
        self.gamma = 0.99  # 0.975
        self.update_target_every = 100
        self.episode_buffer_size = 256
        self.episodes = 50 * 1000

        self.logs_dir = 'logs'
        self.models_save_dir = 'saves'
        self.date_stamp = self.make_date_stamp()

        self.log_dir = None
        self.model_save_dir = None
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
            exploration_decay=self.exploration_decay,
            exploration_min=self.exploration_min,
            exploration_max=self.exploration_max,
            gamma=self.gamma
        )

    @staticmethod
    def make_date_stamp():
        return Datetime.now().strftime("%y%m%d-%H%M")

    @staticmethod
    def kwargify(**kwargs):
        return kwargs

    def init_tensorboard(self):
        dir_save_name = f'{self.name}-{self.date_stamp}'
        self.log_dir = f'{self.logs_dir}/{dir_save_name}'
        self.model_save_dir = f'{self.models_save_dir}/{dir_save_name}'

        train_path = self.log_dir + '/training'
        valid_path = self.log_dir + '/validation'
        self.t_logs_writer = tf.summary.create_file_writer(train_path)
        self.v_logs_writer = tf.summary.create_file_writer(valid_path)

    def train(self):
        state = self.env.reset()
        state = np.expand_dims(state, axis=0)
        most_recent_scores = deque(maxlen=self.episode_buffer_size)
        best_score = -float('inf')
        ma_score = 0

        # fill up memory before training starts
        while self.agent.memory.length() < self.episode_buffer_size:
            action = self.agent.act(state)
            next_state, reward, done, game_info = self.env.step(
                self.env.action_space[action]
            )

            # Change state shape from (Height, Width) to (Height, Width, 1)
            next_state = np.expand_dims(next_state, axis=0)
            self.agent.remember(Transition(
                state=state, action=action, reward=reward,
                next_state=next_state, done=done
            ))
            state = next_state

        pbar = tqdm(range(self.episodes))

        for e in pbar:
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            step = 0

            while not done:
                action = self.agent.act(state)
                step_result = self.env.step(self.env.action_space[action])
                next_state, reward, done, game_info = step_result
                next_state = np.expand_dims(next_state, axis=0)
                self.agent.remember(Transition(
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=done
                ))

                state = next_state
                step += 1

                if step % 10 == 0:
                    pbar.set_description(
                        f"[{self.date_stamp}] "
                        f"Step: {step}. -- ma_score: {ma_score:.4f}"
                    )

            game_score = self.env.get_score()
            most_recent_scores.append(game_score)
            ma_score = np.array(most_recent_scores).mean()
            loss = self.agent.replay(episode_no=e)

            episode_kills = self.env.count_player_kills()
            kill_score = episode_kills
            is_alive = self.env.is_player_alive()
            live_tag = 'alive' if is_alive else 'dead'
            if not is_alive:
                kill_score -= 1

            if e % self.update_target_every == 0:
                self.agent.update_target_model()
                print('TARGET MODEL_UPDATED')

            self.write_logs(
                file_writer=self.t_logs_writer, episode_no=e,
                loss_value=loss, kill_score=kill_score, is_alive=is_alive,
                game_duration=self.env.steps, kills=self.env.player_kills,
                boxes_destroyed=self.env.player_boxes_destroyed,
                score=game_score, ma_score=ma_score
            )

            print(
                f"Episode {e}/{self.episodes - 1} completed "
                f"with {step} steps. "
                f"LR: {self.agent.learning_rate:.5f}. "
                f"EP: {self.agent.exploration_rate:.5f}. "
                f"Kills: {episode_kills} [{live_tag}] "
                f"boxes: {self.env.player_boxes_destroyed} "
                f"score: {game_score:.3f} "
            )

            if ma_score > best_score:
                best_score = ma_score
                score_tag = f'{ma_score:.4f}'.replace('.', '_')
                save_path = f'{self.model_save_dir}/{e}-S{score_tag}.h5'
                print('new best model saved to:', save_path)
                self.agent.save(save_path)

    @staticmethod
    def write_logs(
        file_writer, episode_no: int,
        loss_value: float, kill_score: float, is_alive: int,
        game_duration: int, kills: int, boxes_destroyed: int,
        score: float, ma_score: float
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
            tf.summary.scalar(
                'ma_score', data=ma_score, step=episode_no
            )