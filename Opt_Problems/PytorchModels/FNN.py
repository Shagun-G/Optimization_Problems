import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_dim:int, hidden_layers:list[int], output_dim : int, activation : torch.nn) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation

        self.Layers = nn.ModuleList()

        self.Layers.append(nn.Linear(out_features=self.hidden_layers[0], in_features=self.input_dim))
        for index in range(len(self.hidden_layers) - 1):
            self.Layers.append(nn.Linear(out_features=self.hidden_layers[index + 1], in_features=self.hidden_layers[index]))

        self.Layers.append(nn.Linear(out_features=self.output_dim, in_features=self.hidden_layers[-1]))


    def forward(self, x : torch.Tensor):
        for layer in self.Layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.Layers[-1](x)
        return x