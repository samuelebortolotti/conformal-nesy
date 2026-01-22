from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch


class MNISTLoader:
    def __init__(self, root="./data", val_split=0.1, download=True, active=False):
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.root = root
        self.val_split = val_split
        self.download = download
        self.active = active

    def _label_aggregator(label1):
        return label1

    def load(self):
        full_train = datasets.MNIST(
            self.root, train=True, download=self.download, transform=self.transform
        )
        test = datasets.MNIST(
            self.root, train=False, download=self.download, transform=self.transform
        )

        val_size = int(self.val_split * len(full_train))
        train_size = len(full_train) - val_size
        train, val = random_split(full_train, [train_size, val_size])

        input_dim = (1, 28, 28)
        concept_dim = 10
        output_dim = 10
        n_images = 1
        class_names = [str(i) for i in range(output_dim)]
        concept_names = [str(i) for i in range(concept_dim)]

        return (
            train,
            val,
            test,
            input_dim,
            concept_dim,
            output_dim,
            n_images,
            class_names,
            concept_names,
            lambda label1: label1,
            torch.nn.NLLLoss(),
        )
