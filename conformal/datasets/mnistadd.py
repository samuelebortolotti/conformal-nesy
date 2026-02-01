from torch.utils.data import Dataset
import torch
import random
import numpy as np

from conformal.datasets.mnist import MNISTLoader
from conformal.utils.logic import HardLogic


class MNISTAdditionDataset(Dataset):
    def __init__(self, base_dataset, filter_fn=None):
        self.dataset = base_dataset
        self.filter_fn = filter_fn

        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        self.shuffled_indices = indices
        self.pairs = self._build_pairs()

    def _label_aggregator(self, label1, label2):
        """Aggregate two labels into a single label."""
        return label1 + label2

    def _build_pairs(self):
        pairs = []
        for i in range(len(self.dataset)):
            img1, label1 = self.dataset[i]
            img2, label2 = self.dataset[self.shuffled_indices[i]]

            if self.filter_fn is not None and not self.filter_fn(label1, label2):
                continue

            img_pair = torch.cat([img1, img2], dim=2)  # concatenate along width
            digits = torch.tensor([label1, label2], dtype=torch.long)
            sum_label = self._label_aggregator(label1, label2)

            pairs.append((img_pair, digits, sum_label))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class MNISTAdditionLoader(MNISTLoader):
    def load(self):
        base_train, base_val, base_test, _, _, _, _, _, _, _, _ = super().load()

        input_dim = (1, 28, 56)
        output_dim = 19
        concept_dim = 10
        n_images = 2

        class_names = [str(i) for i in range(output_dim)]
        concept_names = [str(i) for i in range(concept_dim)]

        logic = HardLogic(
            lambda x: x[:, 0] + x[:, 1], n_concepts=n_images, concept_dim=concept_dim
        )

        self.label_weights, self.concept_weights = mnist_addition_weights(
            MNISTAdditionDataset(base_train), output_dim, concept_dim, self.device
        )

        return (
            MNISTAdditionDataset(base_train),
            MNISTAdditionDataset(base_val),
            MNISTAdditionDataset(base_test),
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


def mnist_addition_weights(train_ds, label_count, digit_count, device):
    # Calculate Label Weights
    train_labels = np.array([p[2] for p in train_ds.pairs])
    label_counts = np.bincount(train_labels, minlength=label_count)

    label_weights = torch.tensor(
        [
            len(train_labels) / (float(label_count) * count) if count > 0 else 1.0
            for count in label_counts
        ],
        dtype=torch.float32,
    ).to(device)

    # Calculate Concept Weights
    train_concepts = np.array([p[1].numpy() for p in train_ds.pairs]).flatten()
    concept_counts = np.bincount(train_concepts, minlength=digit_count)

    c_weights_shared = torch.tensor(
        [
            len(train_concepts) / (float(digit_count) * count) if count > 0 else 1.0
            for count in concept_counts
        ],
        dtype=torch.float32,
    ).to(device)
    concept_weights = [c_weights_shared, c_weights_shared]

    return label_weights, concept_weights
