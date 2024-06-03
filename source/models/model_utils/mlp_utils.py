import hydra
import omegaconf
from torch import nn


class MLP(nn.Sequential):
    """Multi-layer perceptron (MLP) module.
    
    Attributes:
        channels: List of channel dimensions for each layer. The first element is the input dimension.
        norm: Specifications for the normalization layer. Expects a dictionary with the key '_target_'.
        act: Specifications for the activation layer. Expects a dictionary with the key '_target_'.
        dropout: Dropout probability. If None, no dropout is applied.
        last_bias: Whether to add a bias to the last linear layer.
    """

    def __init__(self, hidden_channels: list[int], output_dim: int, act: omegaconf.DictConfig,
                 norm: omegaconf.DictConfig | None, dropout: float, last_bias: bool):
        super().__init__()

        if output_dim <= 0:
            raise ValueError(f'Please set the output_dim, got {output_dim}.')

        self.channels = hidden_channels + [output_dim]

        if len(self.channels) <= 2:
            raise ValueError('MLP should have at least 2 layers.')

        for i, channel_dim in enumerate(self.channels[1:-1]):
            self.add_module(f'linear{i}', nn.Linear(self.channels[i], channel_dim))
            if norm:
                self.add_module(f'norm{i}', hydra.utils.instantiate(norm, channel_dim))
            self.add_module(f'act{i}', hydra.utils.instantiate(act, channel_dim))
            if dropout:
                self.add_module(f'dropout{i}', nn.Dropout(dropout))

        self.add_module(f'linear{len(self.channels) - 2}',
                        nn.Linear(self.channels[-2], self.channels[-1], bias=last_bias))
