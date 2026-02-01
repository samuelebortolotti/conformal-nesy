import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
from medmnist import DermaMNIST
from torchvision import transforms
from conformal.utils.logic import HardLogic


class DERMALoader:
    def __init__(
        self,
        data_dir="data",
        batch_size=64,
        device="cuda",
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device
        self.label_weights = []
        self.concepts_weights = []

        # Define the 7 classes as concepts
        self.concept_cols = [
            "Actinic Keratoses",
            "Basal Cell Carcinoma",
            "Benign Keratosis",
            "Dermatofibroma",
            "Melanocytic Nevi",
            "Melanoma",
            "Vascular lesions",
        ]

    def _get_weights(self, dataset):
        # Extract all labels to calculate class weights
        labels = []
        for i in range(len(dataset)):
            # dataset[i] returns (img, concept_vec, label)
            labels.append(dataset[i][2].item())

        labels = np.array(labels)
        counts = np.bincount(labels)
        weights = torch.tensor(len(labels) / (len(counts) * counts), dtype=torch.float)
        return weights.to(self.device)

    def _get_concept_weights(self, dataset):
        # DermaMNIST concepts are mutually exclusive (one-hot)
        # We calculate frequency for each of the 7 slots
        concepts_list = []
        for i in range(len(dataset)):
            concepts_list.append(dataset[i][1].numpy())

        concepts_matrix = np.array(concepts_list)
        weights_list = []
        for i in range(concepts_matrix.shape[1]):
            counts = np.bincount(concepts_matrix[:, i].astype(int))
            # Handle cases where a concept might not appear in a subset
            if len(counts) < 2:
                w = [1.0, 1.0]
            else:
                w = len(concepts_matrix) / (2.0 * counts)
            weights_list.append(torch.tensor(w, dtype=torch.float).to(self.device))
        return weights_list

    def load(self):
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_ds = DERMAMINST(root=self.data_dir, split="train", transform=transform)
        val_ds = DERMAMINST(root=self.data_dir, split="val", transform=transform)
        test_ds = DERMAMINST(root=self.data_dir, split="test", transform=transform)

        self.label_weights = self._get_weights(train_ds)
        self.concepts_weights = self._get_concept_weights(train_ds)

        logic = HardLogic(
            lambda x: (np.any(x[:, [0, 1, 5]] == 1, axis=1)).astype(np.int64),
            n_concepts=1,
            concept_dim=7,
        )

        return (
            train_ds,
            val_ds,
            test_ds,
            (3, 224, 224),  # Input shape
            7,  # n_concepts
            2,  # n_classes
            1,  # logic output dim
            ["Benign", "Malignant"],
            self.concept_cols,
            logic,
            torch.nn.CrossEntropyLoss(weight=self.label_weights),
            self.concepts_weights,
            self.label_weights,
        )


class DERMAMINST(Dataset):
    """Internal helper to format DermaMNIST items correctly for the Loader."""

    def __init__(self, root, split, transform):
        # Force RGB conversion inside the wrapper
        self.base_dataset = DermaMNIST(root=root, download=True, split=split, size=224)
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        class_id = int(target)

        # Target Logic: 0, 1, 5 are Malignant (1), others are Benign (0)
        label = 1 if class_id in [0, 1, 5] else 0

        # One-hot encode the 7 possible classes as 'concepts'
        concepts = torch.zeros(7)
        concepts[class_id] = 1

        return img, concepts, torch.tensor(label, dtype=torch.long)
