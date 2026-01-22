from torch.utils.data import Dataset
import torch
import random

from conformal.datasets.mnist import MNISTLoader
from conformal.utils.logic import Logic


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

        logic = Logic(
            lambda x: x[:, 0] + x[:, 1],
            n_concepts=n_images, 
            concept_dim=concept_dim
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
            torch.nn.NLLLoss(),
        )
