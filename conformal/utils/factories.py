import torch.optim as optim
from conformal.models.resnet18 import ResNet18
from conformal.models.lenet import LeNet
from conformal.models.linear import Linear
from conformal.datasets.mnist import MNISTLoader
from conformal.datasets.mnistadd import MNISTAdditionLoader
from conformal.datasets.mnistsump import MNISTSumParityLoader
from conformal.datasets.mnisthalf import MNISTHalfLoader
from conformal.datasets.cub import CUBLoader
from conformal.datasets.boia import BOIALoader
from conformal.datasets.chx import CHXLoader
from conformal.models.dpl import DPL
from conformal.models.ltn import LTN


class OptimizerFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_optimizer(optimizer_type, model_parameters, lr=1e-3, momentum=0.9):
        if optimizer_type == "adam":
            return optim.Adam(model_parameters, lr=lr)
        elif optimizer_type == "sgd":
            return optim.SGD(model_parameters, lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class NetworkFactory:
    @staticmethod
    def get_network(name: str, input_shape=(1, 28, 28), output_dim=10, args=None):
        if name.lower() == "resnet18":
            return ResNet18(input_shape=input_shape, num_classes=output_dim)
        elif name.lower() == "lenet":
            return LeNet(input_shape=input_shape, num_classes=output_dim)
        elif name.lower() == "linear":
            return Linear(input_shape=input_shape, num_classes=output_dim)
        else:
            raise ValueError(f"Unknown network type: {name}")


class DatasetFactory:
    @staticmethod
    def get_dataset(name: str, **kwargs):
        name = name.lower()
        # if name == "mnist":
        #     return MNISTLoader(**kwargs).load()
        if name == "mnistadd":
            return MNISTAdditionLoader(**kwargs).load()
        elif name == "mnistsump":
            return MNISTSumParityLoader(**kwargs).load()
        elif name == "mnisthalf":
            return MNISTHalfLoader(**kwargs).load()
        # elif name == "cub":
        #     return CUBLoader(**kwargs).load()
        elif name == "boia":
            return BOIALoader(**kwargs).load()
        elif name == "chx":
            return CHXLoader(**kwargs).load()
        else:
            raise ValueError(f"Unknown dataset: {name}")


class NeSyFactory:
    @staticmethod
    def get_nesy_model(
        name: str, n_images, model, entangled, concept_dim, output_dim, dataset, device
    ):
        if name.lower() == "dpl":
            return DPL(
                n_images, model, entangled, concept_dim, output_dim, dataset, device
            )
        elif name.lower() == "ltn":
            return LTN(
                n_images, model, entangled, concept_dim, output_dim, dataset, device
            )
