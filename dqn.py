import os
from typing import Tuple, Optional

import numpy as np
import random
import torch

from torch import optim, nn
from Transition import Transition
from models.DQN_3D import DQN_3D
from models.SimpleDQN import SimpleDQN
from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from ReplayBuffer import ReplayMemory


class DQN:
    def __init__(
        self, state_shape, action_size: int,
        learning_rate_max=0.001, gamma=0.75, memory_size=600,
        batch_size=32, exploration_max=1.0, exploration_min=0.01,
        exploration_decay=0.995, use_gpu: bool = True
    ):
        self.state_shape = state_shape
        self.state_tensor_shape = (-1,) + state_shape
        self.action_size = action_size
        self.learning_rate_max = learning_rate_max
        self.learning_rate = learning_rate_max
        self.memory_size = memory_size
        self.gamma = gamma

        self.memory = PrioritizedReplayBuffer(capacity=self.memory_size)

        self.batch_size = batch_size
        self.exploration_rate = exploration_max
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.eval()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.update_target_model()

    def _build_model(self):
        """
        model = DQN_3D(
            fcc_input_size=8 * 13 ** 2, num_actions=self.action_size
        ).to(self.device)
        """
        model = SimpleDQN(
            fcc_input_size=11 * 13 ** 2, num_actions=self.action_size
        ).to(self.device)
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, transition: Transition):
        if transition.q_values is None:
            q_values = self.get_q_values(transition.state)
        else:
            q_values = transition.q_values

        q_value = q_values[0, transition.action]

        if transition.next_q_values is None:
            next_q_values = self.get_q_values(transition.next_state)
        else:
            next_q_values = transition.q_values

        max_q_value = next_q_values.max(1)[0]
        error = (q_value - (transition.reward + self.gamma * max_q_value)) ** 2
        self.memory.push(transition, priority=error)

    def act(self, state, illegal_actions=None, epsilon=None) -> int:
        return self.get_action(
            state, illegal_actions=illegal_actions, epsilon=epsilon
        )[0]

    def get_action(
        self, state, illegal_actions=None, epsilon=None
    ) -> Tuple[int, Optional[np.ndarray]]:
        if epsilon is None:
            epsilon = self.exploration_rate
        if illegal_actions is None:
            illegal_actions = []

        if np.random.rand() < epsilon:
            actions = np.arange(self.action_size)
            # MUST change to float as np.nan is float type
            actions = actions.astype(float)
            # Set illegal actions to NaN
            actions[illegal_actions] = np.nan
            # '~' character is to filter out NaN elements
            legal_actions = actions[~np.isnan(actions)]

            # return random.randrange(self.action_size)
            return int(random.choice(legal_actions)), None

        q_values = self.get_q_values(state, illegal_actions)
        # Return argmax ignoring NaNs.
        return np.nanargmax(q_values[0]), q_values

    def get_q_values(self, state, illegal_actions=None):
        if illegal_actions is None:
            illegal_actions = []

        state = torch.tensor(state).to(self.device)
        with torch.no_grad():
            raw_prediction = self.model.forward(state)

        prediction = raw_prediction.cpu().numpy()
        prediction[0][illegal_actions] = np.nan
        return prediction

    def get_max_q_value(self, state, illegal_actions=None):
        q_values = self.get_q_values(state, illegal_actions=illegal_actions)
        return q_values.max(axis=1)[0]

    def replay(self, episode_no: int = 0):
        if self.memory.length() < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        batch_size = len(batch)

        state_batch = torch.tensor(np.concatenate(batch.states))
        state_batch = state_batch.to(torch.float32).to(self.device)
        action_batch = torch.tensor(batch.actions).to(torch.int64)
        action_batch = action_batch.to(self.device).unsqueeze(0)
        reward_batch = torch.tensor(batch.rewards)
        reward_batch = reward_batch.to(torch.float32).to(self.device)
        dones_batch = torch.tensor(batch.dones).to(torch.int32)
        non_final_indexes = [
            k for k in range(batch_size) if dones_batch[k] == 0
        ]

        next_state_batch = torch.tensor(np.concatenate(batch.next_states))
        next_state_batch = next_state_batch.to(torch.float32).to(self.device)
        non_final_next_states = next_state_batch[non_final_indexes]
        non_final_next_states = non_final_next_states.to(torch.float32)

        # get Q-value predictions for state-action pairs that were taken
        q_values_batch = self.model.forward(state_batch)
        state_action_values = torch.gather(q_values_batch, 1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self.device)

        if len(non_final_next_states) > 0:
            with torch.no_grad():
                # assign q-values of next states for non-final states
                # i.e. for states that haven't ended
                next_state_values[non_final_indexes] = self.target_model.forward(
                    non_final_next_states
                ).max(1).values

        expected_state_action_values = (
            (next_state_values * self.gamma) + reward_batch
        )

        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, expected_state_action_values)
        loss_value = loss.item()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(
            self.model.parameters(), 100
        )
        self.optimizer.step()

        self.exploration_rate = (
            self.exploration_max * self.exploration_decay ** episode_no
        )
        self.exploration_rate = max(
            self.exploration_min, self.exploration_rate
        )
        return loss_value

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.target_model.load_state_dict(torch.load(name))

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        torch.save(self.model.state_dict(), path)

