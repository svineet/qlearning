import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    """
        Neural network that estimates Q value of given
        the state and action vectors
    """
    def __init__(self, state_size, action_size, H1, H2):
        self.model = nn.Sequential(
                nn.Linear(state_size+action_size, H1), nn.ReLU(inplace=True),
                nn.Linear(H1, H2), nn.ReLU(inplace=True),
                nn.Linear(H2, 1)
        )

    def forward(self, x):
        return self.model(x)

