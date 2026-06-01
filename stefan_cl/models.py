"""Networks: the task classifier and the frontier signed-distance field."""
import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):
    """2-hidden-layer ReLU MLP. Maps R^2 -> 2 logits (binary task labels)."""

    def __init__(self, in_dim=2, hidden=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FrontierField(nn.Module):
    """Signed-distance field phi(x). 3-hidden-layer tanh MLP -> scalar.
    Tanh activations give smooth, well-defined gradients for the Eikonal term.
    Negative inside the consolidated region, positive outside, zero on Gamma."""

    def __init__(self, in_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
