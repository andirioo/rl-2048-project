"""Student extension B: network architecture. The algorithm interprets outputs as Q-values or policy logits."""
import torch
from torch import nn
from encoders import input_channels


class MLP(nn.Module):
    def __init__(self, channels, hidden, output_dim=4):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(channels * 16, hidden),
                                    nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, output_dim))

    def forward(self, x):
        return self.layers(x)


class BoardCNN(nn.Module):
    """Horizontal/vertical convolutions with six feature branches, shared by training and evaluation."""
    def __init__(self, channels, depth, hidden, output_dim=4):
        super().__init__()
        self.horizontal1 = nn.Conv2d(channels, depth, (1, 2))
        self.vertical1 = nn.Conv2d(channels, depth, (2, 1))
        self.horizontal2 = nn.Conv2d(depth, depth, (1, 2))
        self.vertical2 = nn.Conv2d(depth, depth, (2, 1))
        # 12+12+8+9+9+8 spatial positions, each with depth features.
        self.head = nn.Sequential(nn.Linear(58 * depth, hidden), nn.ReLU(),
                                  nn.Linear(hidden, output_dim))

    def forward(self, x):
        h = torch.relu(self.horizontal1(x))
        v = torch.relu(self.vertical1(x))
        features = [h, v, torch.relu(self.horizontal2(h)),
                    torch.relu(self.vertical2(h)), torch.relu(self.horizontal2(v)),
                    torch.relu(self.vertical2(v))]
        return self.head(torch.cat([f.flatten(1) for f in features], dim=1))


def build_model(config, output_dim=4):
    # STUDENT EDIT B: Register your network name here.
    channels = input_channels(config)
    if config.network == "cnn":
        model = BoardCNN(channels, config.conv_depth, config.hidden_units, output_dim)
    elif config.network == "mlp":
        model = MLP(channels, config.hidden_units, output_dim)
    else:
        raise ValueError(f"Unknown network {config.network}; register it in models.build_model")
    # Check shapes before starting training.
    with torch.no_grad():
        output = model(torch.zeros(2, channels, 4, 4))
    if output.shape != (2, output_dim):
        raise ValueError(f"Expected network output (batch, {output_dim}); got {output.shape}")
    return model
