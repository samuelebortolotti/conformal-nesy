import torch.nn as nn
import torchvision.models as models

from conformal.models import dpl, ltn, linear_predictor, dsl


def configure_global_arguments(parser):
    """Configure global arguments that are shared across models and datasets."""
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for ResNet-18.",
    )


class ResNet18(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, pretrained=True):
        super().__init__()
        self.input_shape = input_shape
        base_model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        if input_shape[0] == 1:
            base_model.conv1 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            base_model.maxpool = nn.Identity()

        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)

        self.model = base_model

        # initialize the weights randomly
        if not pretrained:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for ResNet18
    resnet_parser = subparsers.add_parser(
        "resnet18",
        help="Train a ResNet-18 model",
    )

    configure_global_arguments(resnet_parser)

    subparsers = resnet_parser.add_subparsers(dest="nesy")
    ltn.configure_subparsers(subparsers)
    dpl.configure_subparsers(subparsers)
    linear_predictor.configure_subparsers(subparsers)
    dsl.configure_subparsers(subparsers)
