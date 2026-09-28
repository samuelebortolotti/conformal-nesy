import torch
import torch.nn as nn
from conformal.models import dpl, ltn, linear_predictor, dsl


def configure_global_arguments(parser):
    pass


class LeNet(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        super().__init__()
        self.input_shape = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.flatten_dim = self._get_flatten_dim(input_shape)

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def _get_flatten_dim(self, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            return out.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for LeNet5
    lenet_parser = subparsers.add_parser(
        "lenet",
        help="Train a LeNet model",
    )

    configure_global_arguments(lenet_parser)

    subparsers = lenet_parser.add_subparsers(dest="nesy")
    ltn.configure_subparsers(subparsers)
    dpl.configure_subparsers(subparsers)
    linear_predictor.configure_subparsers(subparsers)
    dsl.configure_subparsers(subparsers)
