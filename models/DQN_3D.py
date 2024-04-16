import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple, Iterable, List


class DQN_3D(nn.Module):
    def __init__(
        self, fcc_input_size: int,
        fcn_layers=(128, 32, 16), num_actions: int = 6,
        dropout_p: float = 0.0, use_batch_norm: bool = False
    ):
        super(DQN_3D, self).__init__()
        self.fcc_list = fcn_layers + (num_actions,)
        self.use_batch_norm = use_batch_norm
        self.fcc_input_size = fcc_input_size
        self.num_actions = num_actions

        # Convolutional layers to extract features
        self.conv_layers = nn.Sequential(
            nn.Conv3d(
                in_channels=1, out_channels=32,
                stride=1, padding=1, kernel_size=(11, 5, 5)
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=32, out_channels=8, kernel_size=3,
                stride=1, padding=1
            ),
            nn.ReLU()
        )

        # Fully connected layers to output action Q-values
        self.fc_layers = nn.Sequential(
            nn.Linear(2904, 256),
            nn.ReLU(),
            # Output action Q-values for 6 actions
            nn.Linear(256, self.num_actions)
        )

    @staticmethod
    def make_dense(
        fcn_layers: Tuple[int, ...], input_size: int = 2048,
        dropout_p: float = 0.0, num_outputs: int = 1,
        use_batch_norm: bool = False
    ):
        num_neurons = fcn_layers[0]
        dense_layers = [
            nn.Linear(input_size, fcn_layers[0]),
        ]

        if dropout_p > 0.0:
            dense_layers.append(nn.Dropout(dropout_p))

        dense_layers.append(nn.ReLU())

        for k in range(1, len(fcn_layers)):
            num_neurons = fcn_layers[k]
            prev_neurons = fcn_layers[k - 1]
            dense_layers.append(nn.Linear(prev_neurons, num_neurons))

            if use_batch_norm:
                dense_layers.append(nn.BatchNorm1d(num_neurons))
            if dropout_p > 0.0:
                dense_layers.append(nn.Dropout(dropout_p))

            dense_layers.append(nn.ReLU())

        dense_sequential = nn.Sequential(
            *dense_layers,
            nn.Linear(num_neurons, num_outputs)
        )

        return dense_sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.unsqueeze(1)
        # batch, channels, height, width
        assert len(shape) == 4
        conv_out = self.conv_layers(x)
        x = conv_out.reshape(shape[0], -1)
        x = self.fc_layers(x)
        return x