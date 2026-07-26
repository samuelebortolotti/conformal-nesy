import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from conformal.models import resnet18, lenet, linear, clip_encoder
from conformal.utils.logic import HardLogic
from conformal.datasets.rival import (
    RIVAL10Loader,
    CIFAR_CONCEPT_LIST,
    CIFAR_CLASS_NAMES,
    CIFAR_CLASS_TO_CONCEPTS,
)
from torchvision.datasets import CIFAR10


def configure_global_arguments(parser):
    """Global arguments for CIFAR10"""
    pass


class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.cifar_ds = CIFAR10(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.cifar_ds)

    def get_concept(self, label):
        concepts = torch.zeros(len(CIFAR_CONCEPT_LIST), dtype=torch.long)
        for c in CIFAR_CLASS_TO_CONCEPTS[label]:
            concepts[CIFAR_CONCEPT_LIST.index(c)] = 1
        return concepts

    def __getitem__(self, idx):
        img, cifar_label = self.cifar_ds[idx]
        if self.transform:
            img = self.transform(img)
        concepts = self.get_concept(CIFAR_CLASS_NAMES[cifar_label])

        return img, concepts, cifar_label


class CIFAR10Loader(RIVAL10Loader):
    def load(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                ),
            ]
        )

        full_train_ds = CIFAR10Dataset(
            root=self.data_dir, train=True, transform=transform
        )
        test_ds = CIFAR10Dataset(root=self.data_dir, train=False, transform=transform)

        indices = list(range(len(full_train_ds)))
        train_idx, val_idx = train_test_split(
            indices, test_size=self.val_split, random_state=42
        )

        train_ds = torch.utils.data.Subset(full_train_ds, train_idx)
        val_ds = torch.utils.data.Subset(full_train_ds, val_idx)

        train_labels = []
        train_concepts = []

        for idx in train_idx:
            _, c, y = full_train_ds[idx]
            train_labels.append(y)
            train_concepts.append(c.numpy())

        train_labels = np.array(train_labels)
        train_concepts = np.array(train_concepts)

        # Label weights (CrossEntropy)
        y_counts = np.bincount(train_labels, minlength=self.n_labels)
        self.label_weights = torch.tensor(
            len(train_labels) / (self.n_labels * y_counts), dtype=torch.float
        ).to(self.device)

        # Concept weights (BCEWithLogitsLoss or specific concept loss)
        for i in range(self.concept_dim):
            c_counts = np.bincount(train_concepts[:, i].astype(int), minlength=2)
            if c_counts[0] == 0 or c_counts[1] == 0:
                weights = [1.0, 1.0]
            else:
                weights = len(train_labels) / (2.0 * c_counts)
            self.concepts_weights.append(
                torch.tensor(weights, dtype=torch.float).to(self.device)
            )

        logic = HardLogic(
            self._return_cifar_logic(),
            n_concepts=self.concept_dim,
            concept_dim=2,
            multi_set_logic=self._return_multi_set_cifar_logic(),
        )

        return (
            train_ds,
            val_ds,
            test_ds,
            (3, 32, 32),
            self.concept_dim,
            self.n_labels,
            1,
            CIFAR_CLASS_NAMES,
            CIFAR_CONCEPT_LIST,
            logic,
            torch.nn.CrossEntropyLoss(weight=self.label_weights),
            self.concepts_weights,
            self.label_weights,
        )


def configure_subparsers(subparsers):
    rival_parser = subparsers.add_parser(
        "cifar",
        help="Use CIFAR10 dataset",
    )

    configure_global_arguments(rival_parser)

    sub = rival_parser.add_subparsers(dest="model")
    resnet18.configure_subparsers(sub)
    lenet.configure_subparsers(sub)
    linear.configure_subparsers(sub)
    clip_encoder.configure_subparsers(sub)
