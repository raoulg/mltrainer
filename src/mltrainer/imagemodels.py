from dataclasses import asdict, dataclass

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(
        self, features: int, num_classes: int, units1: int, units2: int
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.units1 = units1
        self.units2 = units2
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(features, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CNN(nn.Module):
    def __init__(
        self,
        features: int,
        num_classes: int,
        kernel_size: int,
        filter1: int,
        filter2: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.filter1 = filter1
        self.filter2 = filter2

        self.convolutions = nn.Sequential(
            nn.Conv2d(features, filter1, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filter1, filter2, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filter2, 32, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        logits = self.dense(x)
        return logits


@dataclass
class CNNConfig:
    matrixshape: tuple
    batchsize: int
    input_channels: int
    hidden: int
    kernel_size: int
    maxpool: int
    num_layers: int
    num_classes: int


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class CNNblocks(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = asdict(config)
        input_channels = config.input_channels
        kernel_size = config.kernel_size
        hidden = config.hidden
        # first convolution
        self.convolutions = nn.ModuleList(
            [
                ConvBlock(input_channels, hidden, kernel_size),
            ]
        )

        # additional convolutions
        pool = config.maxpool
        num_maxpools = 0
        for i in range(config.num_layers):
            self.convolutions.extend(
                [ConvBlock(hidden, hidden, kernel_size), nn.ReLU()]
            )
            # every two layers, add a maxpool
            if i % 2 == 0:
                num_maxpools += 1
                self.convolutions.append(nn.MaxPool2d(pool, pool))

        # let's try to calculate the size of the linear layer
        # please note that changing stride/padding will change the logic
        matrix_size = (config.matrixshape[0] // (pool**num_maxpools)) * (
            config.matrixshape[1] // (pool**num_maxpools)
        )
        print(f"Calculated matrix size: {matrix_size}")
        print(f"Caluclated flatten size: {matrix_size * hidden}")

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(matrix_size * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)
        x = self.dense(x)
        return x
