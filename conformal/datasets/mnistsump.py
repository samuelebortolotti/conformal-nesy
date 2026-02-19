from conformal.datasets.mnist import MNISTLoader
from conformal.datasets.mnistadd import MNISTAdditionDataset
from conformal.utils.logic import HardLogic
from conformal.models import resnet18, lenet, linear
from conformal.datasets.mnistadd import mnist_addition_weights
import torch


def configure_global_arguments(parser):
    """Global arguments for MNIST-SumParity"""
    pass


class MNISTSumParityDataset(MNISTAdditionDataset):
    def _label_aggregator(self, label1, label2):
        return (label1 + label2) % 2


class MNISTSumParityLoader(MNISTLoader):
    def load(self):
        base_train, base_val, base_test, _, _, _, _, _, _, _, _ = super().load()

        input_dim = (1, 28, 28)
        output_dim = 2
        concept_dim = 10
        n_images = 2

        class_names = [str(i) for i in range(output_dim)]
        concept_names = [str(i) for i in range(concept_dim)]

        logic = HardLogic(
            lambda x: (x[:, 0] + x[:, 1]) % 2,
            n_concepts=n_images,
            concept_dim=concept_dim,
        )

        self.label_weights, self.concept_weights = mnist_addition_weights(
            MNISTSumParityDataset(base_train), output_dim, concept_dim, self.device
        )

        return (
            MNISTSumParityDataset(base_train),
            MNISTSumParityDataset(base_val),
            MNISTSumParityDataset(base_test),
            input_dim,
            concept_dim,
            output_dim,
            n_images,
            class_names,
            concept_names,
            logic,
            torch.nn.NLLLoss(weight=self.label_weights),
            self.concept_weights,
            self.label_weights,
        )


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for MNIST-Sumparity
    mnist_sump_parser = subparsers.add_parser(
        "mnistsump",
        help="Use MNIST-SumParity as dataset",
    )
    configure_global_arguments(mnist_sump_parser)

    subparsers = mnist_sump_parser.add_subparsers(dest="model")
    resnet18.configure_subparsers(subparsers)
    lenet.configure_subparsers(subparsers)
    linear.configure_subparsers(subparsers)
