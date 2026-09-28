from conformal.datasets.mnist import MNISTLoader
from conformal.datasets.mnistadd import MNISTAdditionDataset
from conformal.utils.logic import HardLogic
from conformal.datasets.mnistadd import mnist_addition_weights
from conformal.models import resnet18, lenet, linear, clip_encoder
import torch


def configure_global_arguments(parser):
    pass


class MNISTEvenOddDataset(MNISTAdditionDataset):
    def _label_aggregator(self, label1, label2):
        return label1 + label2


class MNISTEvenOddLoader(MNISTLoader):
    def load(self):
        def _in_distribution_filter(label1, label2):
            allowed_pairs = {
                (0, 6),
                (2, 8),
                (4, 6),
                (4, 8),
                (1, 5),
                (3, 7),
                (1, 9),
                (3, 9),
            }
            return (label1, label2) in allowed_pairs

        base_train, base_val, base_test, _, _, _, _, _, _, _, _ = super().load()

        input_dim = (1, 28, 28)
        output_dim = 19
        concept_dim = 10
        n_images = 2

        class_names = [str(i) for i in range(output_dim)]
        concept_names = [str(i) for i in range(concept_dim)]

        logic = HardLogic(
            lambda x: x[:, 0] + x[:, 1], n_concepts=n_images, concept_dim=concept_dim
        )

        self.label_weights, self.concept_weights = mnist_addition_weights(
            MNISTEvenOddDataset(base_train, filter_fn=_in_distribution_filter),
            output_dim,
            concept_dim,
            self.device,
        )

        return (
            MNISTEvenOddDataset(base_train, filter_fn=_in_distribution_filter),
            MNISTEvenOddDataset(base_val, filter_fn=_in_distribution_filter),
            MNISTEvenOddDataset(
                base_test,
                filter_fn=_in_distribution_filter,  # _out_of_distribution_filter),
            ),
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
    # Subparser for MNIST-Even-Odd
    mnist_eo_parser = subparsers.add_parser(
        "mnistevenodd",
        help="Use MNIST-Even-Odd as dataset",
    )
    configure_global_arguments(mnist_eo_parser)

    subparsers = mnist_eo_parser.add_subparsers(dest="model")
    resnet18.configure_subparsers(subparsers)
    lenet.configure_subparsers(subparsers)
    linear.configure_subparsers(subparsers)
    clip_encoder.configure_subparsers(subparsers)
