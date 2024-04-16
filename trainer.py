import os
from typing import Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pygame
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from game.Incentives import Incentives
from Transition import Transition
from collections import deque
from datetime import datetime as Datetime
from enums.algorithm import Algorithm
from game.BombermanEnv import BombermanEnv
from TrainingSettings import TrainingSettings
from dqn import DQN


class Trainer(object):

    def __init__(
        self, name='ddqn', incentives: Incentives = Incentives(),
        training_settings: TrainingSettings = TrainingSettings(),
    ):
        self.name = name
        self.incentives = incentives
        self.training_settings = training_settings

        self.learning_rate = training_settings.learning_rate
        self.exploration_decay = training_settings.exploration_decay
        self.exploration_max = training_settings.exploration_max
        self.exploration_min = training_settings.exploration_min
        self.gamma = training_settings.gamma
        self.update_target_every = training_settings.update_target_every
        self.episode_buffer_size = training_settings.episode_buffer_size
        self.episodes = training_settings.episodes
        self.pool_duration = training_settings.pool_duration
        self.verbose = training_settings.verbose

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
            training_settings=self.training_settings
        )

        self.agent = DQN(
            state_shape=self.env.state_shape,
            action_size=self.env.action_space_size,
            batch_size=self.episode_buffer_size,
            learning_rate_max=self.learning_rate,
            exploration_decay=self.exploration_decay,
            exploration_min=self.exploration_min,
            exploration_max=self.exploration_max,
            use_gpu=training_settings.use_gpu,
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

    def get_manual_action_no(self) -> int:
        ####################################################################################
        ####################################################################################
        ''' NOTE: DO NOT REMOVE
            NOTE: Use manual player game control ONLY to check if rewards are truly working
                OR perhaps for pre-training before letting the model choose on its own.

                - Arrow keys to move
                - 'Space' for bomb
                - 'w' for wait
        '''
        ####################################################################################
        action_no = 5
        pygame.event.clear()

        while True:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action_no = self.env.action_space_idx_map[self.env.UP]
                    break
                elif event.key == pygame.K_DOWN:
                    action_no = self.env.action_space_idx_map[self.env.DOWN]
                    break
                elif event.key == pygame.K_LEFT:
                    action_no = self.env.action_space_idx_map[self.env.LEFT]
                    break
                elif event.key == pygame.K_RIGHT:
                    action_no = self.env.action_space_idx_map[self.env.RIGHT]
                    break
                elif event.key == pygame.K_SPACE:
                    action_no = self.env.action_space_idx_map[self.env.BOMB]
                    break
                # else:
                # -- If you wish to not have a choice to wait, but
                # that the AI would auto-wait if there is no input,
                #     then uncomment the "else" line and comment out the "elif" line.
                elif event.key == pygame.K_w:
                    action_no = self.env.action_space_idx_map[self.env.WAIT]
                    break

        return action_no

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def train(self):
        state = self.env.reset()
        state = np.expand_dims(state, axis=0)
        most_recent_scores = deque(maxlen=self.episode_buffer_size)
        best_score = -float('inf')
        ma_score = 0

        simulate_time = self.env.simulate_time
        self.env.simulate_time = True

        # fill up memory before training starts
        while self.agent.memory.length() < self.episode_buffer_size:
            action_no = self.agent.act(
                state, illegal_actions=self.env.get_illegal_actions()
            )

            next_state, reward, done, game_info = self.env.step(
                self.env.action_space[action_no]
            )

            # Change state shape from (Height, Width) to (Height, Width, 1)
            next_state = np.expand_dims(next_state, axis=0)
            self.agent.remember(Transition(
                state=state, action=action_no, reward=reward,
                next_state=next_state, done=done
            ))
            state = next_state

        self.env.simulate_time = simulate_time
        pbar = tqdm(range(self.episodes))

        for e in pbar:
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            step = 0

            pooled_rewards = 0
            last_pooled_step = 0
            last_action_no = None
            pooled_transition: Optional[Transition] = None

            while not done:
                q_values = None

                if not self.training_settings.IS_MANUAL_CONTROL:
                    # Q-learning Model Picking of Action
                    if pooled_transition is None:
                        # continue with previous movement
                        action_no, q_values = self.agent.get_action(
                            state, illegal_actions=self.env.get_illegal_actions()
                        )
                        self.log('ACT', self.env.to_action(action_no))
                    else:
                        last_action = self.env.to_action(last_action_no)
                        assert last_action != self.env.BOMB
                        action_no = last_action_no
                        self.log('REPEAT', self.env.to_action(action_no))
                else:
                    assert self.training_settings.IS_MANUAL_CONTROL
                    action_no = self.get_manual_action_no()

                action = self.env.to_action(action_no)
                if action == self.env.BOMB:
                    pass

                step_result = self.env.step(action)
                next_state, reward, done, game_info = step_result
                next_state = np.expand_dims(next_state, axis=0)

                transition = Transition(
                    state=state, action=action_no, reward=reward,
                    next_state=next_state, done=done,
                    q_values=q_values
                )

                if self.training_settings.POOL_TRANSITIONS:
                    flush = done or (action_no == self.env.BOMB)
                    if flush:
                        self.agent.remember(transition)

                    if pooled_transition is not None:
                        pooled_rewards += reward
                        time_passed = step - last_pooled_step

                        if flush or (time_passed >= self.pool_duration):
                            pooled_transition.next_state = next_state
                            pooled_transition.next_q_values = q_values
                            pooled_transition.reward = pooled_rewards
                            self.agent.remember(pooled_transition)
                            pooled_transition = None

                    elif action != self.env.BOMB:
                        pooled_rewards = reward
                        pooled_transition = transition
                        last_pooled_step = step
                else:
                    self.agent.remember(transition)

                last_action_no = action_no
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