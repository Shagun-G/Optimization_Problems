import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: torch.nn,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation

        self.Layers = nn.ModuleList()
        self.Layers.append(nn.Flatten())
        self.Layers.append(
            nn.Linear(out_features=self.hidden_layers[0], in_features=self.input_dim)
        )
        for index in range(len(self.hidden_layers) - 1):
            self.Layers.append(
                nn.Linear(
                    out_features=self.hidden_layers[index + 1],
                    in_features=self.hidden_layers[index],
                )
            )

        self.Layers.append(
            nn.Linear(out_features=self.output_dim, in_features=self.hidden_layers[-1])
        )

    def forward(self, x: torch.Tensor):
        for layer in self.Layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.Layers[-1](x)
        return x


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x
