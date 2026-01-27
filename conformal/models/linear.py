import torch.nn as nn
from conformal.models import dpl, ltn, linear_predictor, dsl


def configure_global_arguments(parser):
    """Configure global arguments that are shared across models and datasets."""
    pass


class Linear(nn.Module):
    def __init__(self, input_shape=(2048), num_classes=4):
        super().__init__()
        self.input_shape = input_shape

        self.model = nn.Linear(input_shape, num_classes)

    def forward(self, x):
        return self.model(x)


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for Linear
    linear_parser = subparsers.add_parser(
        "linear",
        help="Train a Linear layer",
    )

    configure_global_arguments(linear_parser)

    subparsers = linear_parser.add_subparsers(dest="nesy")
    ltn.configure_subparsers(subparsers)
    dpl.configure_subparsers(subparsers)
    linear_predictor.configure_subparsers(subparsers)
    dsl.configure_subparsers(subparsers)
