import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from collections import namedtuple, deque
from itertools import count
from datetime import datetime as Datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from settings import Settings

"""
Cartpole training example
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Duration should converge to 500+
"""

# To ensure reproducibility during training, you can fix the random seeds
# by uncommenting the lines below. This makes the results consistent across
# runs, which is helpful for debugging or comparing different approaches.
#
# That said, allowing randomness can be beneficial in practice, as it lets
# the model explore different training trajectories.


# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Trainer(object):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.date_stamp = self.make_date_stamp()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.env = gym.make("CartPole-v1")
        # Get number of actions from gym action space\
        # noinspection PyUnresolvedReferences
        self.n_actions = self.env.action_space.n
        # Get the number of state observations
        state, info = self.env.reset()
        n_observations = len(state)

        self.policy_net = DQN(n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.settings.LR,
            amsgrad=True
        )
        self.memory = ReplayMemory(10000)
        self.episode_durations = []
        self.steps_done = 0

        self.log_dir = None
        self.model_save_dir = None
        self.t_logs_writer = None
        self.v_logs_writer = None
        self.init_tensorboard()

    def init_tensorboard(self):
        settings = self.settings
        dir_save_name = f'{settings.name}-{self.date_stamp}'
        self.log_dir = f'{settings.logs_dir}/{dir_save_name}'
        self.model_save_dir = f'{settings.models_save_dir}/{dir_save_name}'

        train_path = self.log_dir + '/training'
        valid_path = self.log_dir + '/validation'
        self.t_logs_writer = tf.summary.create_file_writer(train_path)
        self.v_logs_writer = tf.summary.create_file_writer(valid_path)

    @staticmethod
    def make_date_stamp():
        return Datetime.now().strftime("%y%m%d-%H%M")

    def select_action(self, state):
        settings = self.settings
        sample = random.random()
        eps_threshold = (
            settings.EPS_END + (settings.EPS_START - settings.EPS_END) *
            math.exp(-1. * self.steps_done / settings.EPS_DECAY)
        )
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.env.action_space.sample()]],
                device=self.device, dtype=torch.long
            )

    def optimize_model(self):
        settings = self.settings
        if len(self.memory) < settings.BATCH_SIZE:
            return

        transitions = self.memory.sample(settings.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([
            s for s in batch.next_state if s is not None
        ])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(settings.BATCH_SIZE, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (
            (next_state_values * settings.GAMMA) + reward_batch
        )
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        settings = self.settings

        for episode_no in range(settings.num_episodes):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (
                        policy_net_state_dict[key] * settings.TAU +
                        target_net_state_dict[key] * (1 - settings.TAU)
                    )

                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    # TODO: plot this on tensorboard instead
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

        print('Complete')


if __name__ == '__main__':
    trainer = Trainer(settings=Settings())
    trainer.train()
