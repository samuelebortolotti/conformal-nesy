import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np


class CUBDataset(Dataset):
    """
    Custom dataset for CUB-200-2011 that includes both class labels and 312 concept attributes.
    Each item: (image, concepts, label)
    """

    def __init__(self, root="./data/CUB_200_2011", train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        # Load metadata files
        images_txt = os.path.join(self.root, "images.txt")
        labels_txt = os.path.join(self.root, "image_class_labels.txt")
        split_txt = os.path.join(self.root, "train_test_split.txt")
        attr_txt = os.path.join(self.root, "attributes", "image_attribute_labels.txt")

        # Read image paths and labels
        images_df = pd.read_csv(
            images_txt, sep=" ", header=None, names=["img_id", "filepath"]
        )
        labels_df = pd.read_csv(
            labels_txt, sep=" ", header=None, names=["img_id", "target"]
        )
        split_df = pd.read_csv(
            split_txt, sep=" ", header=None, names=["img_id", "is_train"]
        )

        # Merge
        meta_df = images_df.merge(labels_df, on="img_id").merge(split_df, on="img_id")

        # Filter train/test
        self.meta_df = meta_df[meta_df["is_train"] == int(train)].reset_index(drop=True)

        # Load attributes (image_id, attr_id, is_present, certainty, time)
        attr_df = pd.read_csv(
            attr_txt,
            sep=" ",
            header=None,
            names=["img_id", "attr_id", "is_present", "certainty", "time"],
            on_bad_lines="skip",
        )

        # Pivot to image_id x attr_id matrix
        attr_matrix = (
            attr_df.pivot(index="img_id", columns="attr_id", values="is_present")
            .fillna(0)
            .astype(np.float32)
        )
        self.attr_matrix = attr_matrix

        # Filter to relevant images
        self.meta_df = self.meta_df[self.meta_df["img_id"].isin(attr_matrix.index)]
        self.meta_df = self.meta_df.reset_index(drop=True)

        self.image_dir = os.path.join(self.root, "images")

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filepath"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = int(row["target"]) - 1  # labels are 1–200 → make 0–199

        # Get concepts
        concepts = torch.tensor(
            self.attr_matrix.loc[row["img_id"]].values, dtype=torch.float32
        )

        return img, concepts, label


class CUBLoader:
    """CUB dataset loader with concept (attribute) support, following MNISTLoader structure."""

    def __init__(self, root="./data/CUB_200_2011", val_split=0.1):
        self.root = root
        self.val_split = val_split

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _label_aggregator(self, label):
        return label

    def load(self):
        """Load train/val/test datasets and metadata."""
        full_train = CUBDataset(root=self.root, train=True, transform=self.transform)
        test = CUBDataset(root=self.root, train=False, transform=self.transform)

        # Train/val split
        val_size = int(self.val_split * len(full_train))
        train_size = len(full_train) - val_size
        train, val = random_split(full_train, [train_size, val_size])

        # Dataset metadata
        input_dim = (3, 224, 224)
        concept_dim = 312
        output_dim = 200
        n_images = 1
        class_names = [f"Bird_{i}" for i in range(output_dim)]
        concept_names = [f"Attr_{i}" for i in range(concept_dim)]

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
            self._label_aggregator,
            torch.nn.NLLLoss(),
        )
