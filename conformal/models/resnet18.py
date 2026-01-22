import torch.nn as nn
import torchvision.models as models


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

    def forward(self, x):
        return self.model(x)


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for ResNet18
    resnet_parser = subparsers.add_parser(
        "resnet18",
        help="Train a ResNet-18 model",
    )
    resnet_parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for ResNet-18.",
    )
