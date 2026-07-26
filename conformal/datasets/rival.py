import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from conformal.models import resnet18, lenet, linear, clip_encoder
from conformal.utils.logic import HardLogic

CIFAR_CONCEPT_LIST = [
    "wheels",
    "metallic",
    "wings",
    "animal",
    "hairy",
    "horns",
    "long-snout",
]
CIFAR_CLASS_NAMES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "equine",
    "ship",
    "truck",
]
CIFAR_CLASS_TO_CONCEPTS = {
    "truck": ["wheels", "metallic"],
    "car": ["wheels", "metallic"],
    "plane": ["wings", "metallic"],
    "ship": ["metallic"],
    "cat": ["animal", "hairy"],
    "dog": ["animal", "hairy", "long-snout"],
    "equine": ["animal", "hairy", "long-snout"],
    "deer": ["animal", "hairy", "long-snout", "horns"],
    "frog": ["animal"],
    "bird": ["wings"],
}


def configure_global_arguments(parser):
    """Global arguments for RIVAL10"""
    pass


class RIVAL10Dataset(Dataset):
    def __init__(self, img_files, concepts, labels, transform):
        self.img_files = img_files
        self.transform = transform
        self.concepts = concepts
        self.labels = labels

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        concepts = torch.tensor(self.concepts[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, concepts, label


class RIVAL10Loader:
    def __init__(self, data_dir="data", device="cuda", val_split=0.3):
        self.data_dir = data_dir
        self.device = device
        self.val_split = val_split
        self.concept_dim = len(CIFAR_CONCEPT_LIST)
        self.n_labels = len(CIFAR_CLASS_NAMES)
        self.label_weights = []
        self.concepts_weights = []

    def _return_cifar_logic(self):
        def logic(x):
            """
            Logic function for RIVAL10.
            """
            wheels = x[:, 0]
            metallic = x[:, 1]
            wings = x[:, 2]
            animal = x[:, 3]
            hairy = x[:, 4]
            horns = x[:, 5]
            long_snout = x[:, 6]

            # Pre-calculate common logical conditions
            is_vehicle = (
                metallic * (1 - animal) * (1 - hairy) * (1 - horns) * (1 - long_snout)
            )
            is_living = animal * (1 - metallic) * (1 - wheels)

            truck = is_vehicle * wheels * (1 - wings)
            car = is_vehicle * wheels * (1 - wings)
            plane = is_vehicle * wings
            ship = is_vehicle * (1 - wheels) * (1 - wings)
            bird = is_living * wings * (1 - hairy) * (1 - horns) * (1 - long_snout)
            frog = (
                is_living * (1 - wings) * (1 - hairy) * (1 - horns) * (1 - long_snout)
            )
            deer = is_living * (1 - wings) * hairy * horns * long_snout
            cat = is_living * (1 - wings) * hairy * (1 - horns) * (1 - long_snout)
            dog = is_living * (1 - wings) * hairy * (1 - horns) * long_snout
            equine = is_living * (1 - wings) * hairy * (1 - horns) * long_snout

            preds = np.stack(
                [plane, car, bird, cat, deer, dog, frog, equine, ship, truck], axis=1
            )

            # --------------------------------------------------
            # fallback rules when all zeros
            # --------------------------------------------------

            zero_mask = preds.sum(axis=1) == 0

            for i in np.where(zero_mask)[0]:

                if metallic[i]:
                    preds[i, [0, 1, 9, 8]] = 0.25

                elif wings[i]:
                    preds[i, 0] = 1
                    preds[i, 2] = 1

                elif hairy[i]:
                    preds[i, [3, 4, 5, 7]] = 0.25

                elif long_snout[i]:
                    preds[i, [4, 5, 7]] = 1.0 / 3

                else:
                    preds[i, :] = 1.0 / 10

            return np.argmax(preds, axis=1)

        return logic

    def _return_multi_set_cifar_logic(self):
        def logic(x):
            """
            Logic function for RIVAL10.
            """
            wheels = x[:, 0]
            metallic = x[:, 1]
            wings = x[:, 2]
            animal = x[:, 3]
            hairy = x[:, 4]
            horns = x[:, 5]
            long_snout = x[:, 6]

            # Pre-calculate common logical conditions
            is_vehicle = (
                metallic * (1 - animal) * (1 - hairy) * (1 - horns) * (1 - long_snout)
            )
            is_living = animal * (1 - metallic) * (1 - wheels)

            truck = is_vehicle * wheels * (1 - wings)
            car = is_vehicle * wheels * (1 - wings)
            plane = is_vehicle * wings
            ship = is_vehicle * (1 - wheels) * (1 - wings)
            bird = is_living * wings * (1 - hairy) * (1 - horns) * (1 - long_snout)
            frog = (
                is_living * (1 - wings) * (1 - hairy) * (1 - horns) * (1 - long_snout)
            )
            deer = is_living * (1 - wings) * hairy * horns * long_snout
            cat = is_living * (1 - wings) * hairy * (1 - horns) * (1 - long_snout)
            dog = is_living * (1 - wings) * hairy * (1 - horns) * long_snout
            equine = is_living * (1 - wings) * hairy * (1 - horns) * long_snout

            preds = np.stack(
                [plane, car, bird, cat, deer, dog, frog, equine, ship, truck], axis=1
            )

            # --------------------------------------------------
            # fallback rules when all zeros
            # --------------------------------------------------

            zero_mask = preds.sum(axis=1) == 0

            for i in np.where(zero_mask)[0]:

                if metallic[i]:
                    preds[i, [0, 1, 9, 8]] = 0.25

                elif wings[i]:
                    preds[i, 0] = 1
                    preds[i, 2] = 1

                elif hairy[i]:
                    preds[i, [3, 4, 5, 7]] = 0.25

                elif long_snout[i]:
                    preds[i, [4, 5, 7]] = 1.0 / 3

                else:
                    preds[i, :] = 1.0 / 10

            max_vals = preds.max(axis=1, keepdims=True)
            mask = preds == max_vals

            indices_per_row = [np.where(row)[0] for row in mask]
            return indices_per_row

        return logic

    def process_files(self, files, wnid_to_class, label_mappings):
        paths, concepts, targets = [], [], []
        for f in files:
            if "merged_mask" in f.name:
                continue

            wnid = f.name.split("_")[0]
            if wnid not in wnid_to_class:
                continue

            inet_name = wnid_to_class[wnid]
            class_name, _ = label_mappings[inet_name]

            target_idx = CIFAR_CLASS_NAMES.index(class_name)
            concept_vec = np.zeros(len(CIFAR_CONCEPT_LIST))
            for c in CIFAR_CLASS_TO_CONCEPTS[class_name]:
                concept_vec[CIFAR_CONCEPT_LIST.index(c)] = 1

            paths.append(str(f))
            concepts.append(concept_vec)
            targets.append(target_idx)
        return np.array(paths), np.array(concepts), np.array(targets)

    def load(self):
        meta_dir = Path(self.data_dir) / "RIVAL10" / "meta"
        with open(meta_dir / "label_mappings.json") as f:
            label_mappings = json.load(f)
        with open(meta_dir / "wnid_to_class.json") as f:
            wnid_to_class = json.load(f)

        base_dir = Path(self.data_dir) / "RIVAL10"
        train_files = sorted(list(base_dir.glob("train/ordinary/*.JPEG")))
        test_files = sorted(list(base_dir.glob("test/ordinary/*.JPEG")))

        # Process both sets
        train_paths, train_concepts, train_targets = self.process_files(
            train_files, wnid_to_class, label_mappings
        )
        x_test, c_test, y_test = self.process_files(
            test_files, wnid_to_class, label_mappings
        )

        x_train, x_val, c_train, c_val, y_train, y_val = train_test_split(
            train_paths,
            train_concepts,
            train_targets,
            test_size=self.val_split,
            random_state=42,
        )

        # Label weights for CrossEntropy
        y_counts = np.bincount(y_train, minlength=self.n_labels)
        self.label_weights = torch.tensor(
            len(y_train) / (self.n_labels * y_counts), dtype=torch.float
        ).to(self.device)

        # Concept weights for Multi-label Binary CrossEntropy
        for i in range(c_train.shape[1]):
            c_counts = np.bincount(c_train[:, i].astype(int), minlength=2)
            weights = len(c_train) / (2.0 * c_counts)
            self.concepts_weights.append(
                torch.tensor(weights, dtype=torch.float).to(self.device)
            )

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logic = HardLogic(
            self._return_cifar_logic(),
            n_concepts=self.concept_dim,
            concept_dim=2,
            multi_set_logic=self._return_multi_set_cifar_logic(),
        )

        return (
            RIVAL10Dataset(x_train, c_train, y_train, transform),
            RIVAL10Dataset(x_val, c_val, y_val, transform),
            RIVAL10Dataset(x_test, c_test, y_test, transform),
            (3, 224, 224),
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
        "rival",
        help="Use RIVAL10 dataset",
    )

    configure_global_arguments(rival_parser)

    sub = rival_parser.add_subparsers(dest="model")
    resnet18.configure_subparsers(sub)
    lenet.configure_subparsers(sub)
    linear.configure_subparsers(sub)
    clip_encoder.configure_subparsers(sub)
