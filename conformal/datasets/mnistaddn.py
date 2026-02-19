from torch.utils.data import Dataset
import torch
import random
import numpy as np

from conformal.datasets.mnist import MNISTLoader
from conformal.utils.logic import HardLogic
from conformal.utils.other import int_ge_2
from conformal.models import resnet18, lenet, linear


def configure_global_arguments(parser):
    """Global arguments for MNIST-AddN"""
    parser.add_argument(
        "--n-digits", type=int_ge_2, default="3", help="Number of MNIST-digits"
    )


class MNISTAdditionNDataset(Dataset):
    def __init__(self, base_dataset, n_digits=3, filter_fn=None):
        self.dataset = base_dataset
        self.n_digits = n_digits
        self.filter_fn = filter_fn

        self.shuffled_indices = [
            random.sample(range(len(self.dataset)), len(self.dataset))
            for _ in range(n_digits - 1)
        ]

        self.samples = self._build_samples()

    def _label_aggregator(self, labels):
        return sum(labels)

    def _build_samples(self):
        samples = []

        for i in range(len(self.dataset)):
            imgs = []
            labels = []

            # First digit
            img, label = self.dataset[i]
            imgs.append(img)
            labels.append(label)

            for k in range(self.n_digits - 1):
                img_k, label_k = self.dataset[self.shuffled_indices[k][i]]
                imgs.append(img_k)
                labels.append(label_k)

            if self.filter_fn is not None and not self.filter_fn(*labels):
                continue

            img_concat = torch.cat(imgs, dim=2)

            digits = torch.tensor(labels, dtype=torch.long)
            sum_label = self._label_aggregator(labels)

            samples.append((img_concat, digits, sum_label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def mnist_addition_n_weights(train_ds, label_count, digit_count, n_digits, device):

    train_labels = np.array([p[2] for p in train_ds.samples])
    label_counts = np.bincount(train_labels, minlength=label_count)

    label_weights = torch.tensor(
        [
            len(train_labels) / (float(label_count) * count) if count > 0 else 1.0
            for count in label_counts
        ],
        dtype=torch.float32,
    ).to(device)

    train_concepts = np.array([p[1].numpy() for p in train_ds.samples]).flatten()

    concept_counts = np.bincount(train_concepts, minlength=digit_count)

    c_weights_shared = torch.tensor(
        [
            len(train_concepts) / (float(digit_count) * count) if count > 0 else 1.0
            for count in concept_counts
        ],
        dtype=torch.float32,
    ).to(device)

    concept_weights = [c_weights_shared for _ in range(n_digits)]

    return label_weights, concept_weights


class MNISTAdditionNLoader(MNISTLoader):
    def __init__(self, n_digits=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_digits = n_digits

    def load(self):
        base_train, base_val, base_test, _, _, _, _, _, _, _, _ = super().load()

        input_dim = (1, 28, 28)
        output_dim = 9 * self.n_digits + 1  # possible sums: 0..9*n
        concept_dim = 10
        n_images = self.n_digits

        class_names = [str(i) for i in range(output_dim)]
        concept_names = [str(i) for i in range(concept_dim)]

        logic = HardLogic(
            lambda x: np.sum(x[:, : self.n_digits], axis=1),
            n_concepts=n_images,
            concept_dim=concept_dim,
        )

        train_ds = MNISTAdditionNDataset(base_train, self.n_digits)

        self.label_weights, self.concept_weights = mnist_addition_n_weights(
            train_ds,
            output_dim,
            concept_dim,
            self.n_digits,
            self.device,
        )

        return (
            train_ds,
            MNISTAdditionNDataset(base_val, self.n_digits),
            MNISTAdditionNDataset(base_test, self.n_digits),
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
    # Subparser for MNIST-AddN
    mnist_addn_parser = subparsers.add_parser(
        "mnistaddn",
        help="Use MNIST-AddN as dataset",
    )
    configure_global_arguments(mnist_addn_parser)

    subparsers = mnist_addn_parser.add_subparsers(dest="model")
    resnet18.configure_subparsers(subparsers)
    lenet.configure_subparsers(subparsers)
    linear.configure_subparsers(subparsers)
