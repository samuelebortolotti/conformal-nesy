import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims=(256, 128),
        num_classes=5,
        dropout=0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
