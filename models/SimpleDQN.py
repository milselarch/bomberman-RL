import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple, Iterable, List


class SimpleDQN(nn.Module):
    def __init__(
        self, fcc_input_size: int,
        fcn_layers=(128, 64, 32), num_actions: int = 4,
        dropout_p: float = 0.0, use_batch_norm: bool = False
    ):
        super(SimpleDQN, self).__init__()
        self.fcc_list = fcn_layers + (num_actions,)
        self.use_batch_norm = use_batch_norm
        self.fcc_input_size = fcc_input_size
        self.num_actions = num_actions

        self.fcc = self.make_dense(
            self.fcc_list, input_size=self.fcc_input_size,
            dropout_p=dropout_p, num_outputs=num_actions,
            use_batch_norm=self.use_batch_norm
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
        # batch, channels, height, width
        assert len(shape) == 4
        x = x.reshape((shape[0], -1))
        out = self.fcc(x)
        return out