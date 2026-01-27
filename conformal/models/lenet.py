import torch.nn as nn
from conformal.models import dpl, ltn, linear_predictor, dsl


def configure_global_arguments(parser):
    """Configure global arguments that are shared across models and datasets."""
    pass


class LeNet(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        super().__init__()
        self.input_shape = input_shape

        self.model = nn.Sequential(
            nn.Conv2d(input_shape[0], 6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.model(x)


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
