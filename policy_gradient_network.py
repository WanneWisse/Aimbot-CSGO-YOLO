import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(PolicyNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Skip activation for the last layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(layer_sizes[-1]))  # Log std for output

    def forward(self, x):
        x = self.network(x)
        action_mean = x  # Mean output from last layer
        action_std = torch.exp(self.log_std)  # Standard deviation from log std parameter
        return action_mean, action_std