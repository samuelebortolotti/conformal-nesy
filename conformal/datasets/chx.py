import os
import logging
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from conformal.utils.logic import HardLogic
from conformal.general_utils import log
from conformal.models import resnet18, lenet, linear


def configure_global_arguments(parser):
    """Global arguments for CHX"""
    parser.add_argument(
        "--chx-multi-class",
        action="store_true",
        default=False,
        help="Divide the Abnormal/Healthy into 4 classes",
    )


class CHXDataset(Dataset):
    """Dataset class with lazy loading for Chest X-rays."""

    def __init__(self, image_paths, concepts, targets, transform=None):
        self.image_paths = image_paths
        self.concepts = concepts
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (
            img,
            torch.tensor(self.concepts[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )


class CHXLoader:
    def __init__(
        self,
        data_dir="data",
        batch_size=64,
        test_split=0.2,
        val_split=0.1,
        device="cuda",
        chx_multi_class=False,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.device = device
        self.label_weights = []
        self.concepts_weights = []
        self.chx_multi_class = chx_multi_class

    def _download_and_extract_needed(self, needed_filenames):
        """Downloads NIH tarballs only to extract specific annotated images."""
        img_dir = os.path.join(self.data_dir, "images_nih")
        os.makedirs(img_dir, exist_ok=True)

        links = [
            "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
            "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
            "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
            "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
            "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
            "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
            "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
            "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
            "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
            "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
            "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
            "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
        ]

        for idx, link in enumerate(links):
            # Check if we already have all needed images before downloading next tar
            existing = set(os.listdir(img_dir))
            if needed_filenames.issubset(existing):
                log(
                    "All annotated images already present. Skipping remaining downloads."
                )
                break

            tar_path = os.path.join(self.data_dir, f"batch_{idx+1}.tar.gz")
            log(f"Downloading tarball {idx+1}/12 to find missing annotated images...")
            urllib.request.urlretrieve(link, tar_path)

            with tarfile.open(tar_path, "r:gz") as tar:
                # Extract only the members that are in our 'needed' list
                members = tar.getmembers()
                to_extract = [
                    m for m in members if os.path.basename(m.name) in needed_filenames
                ]

                if to_extract:
                    log(f"Extracting {len(to_extract)} images from batch_{idx+1}...")
                    tar.extractall(path=img_dir, members=to_extract)

            # Delete tarball immediately to save 2GB+ per batch
            os.remove(tar_path)

    def load(self):
        # 1. Access local CSVs
        readers_csv = os.path.join(
            self.data_dir, "four_findings_expert_labels_individual_readers.csv"
        )
        test_labels_csv = os.path.join(
            self.data_dir, "four_findings_expert_labels_test_labels.csv"
        )
        val_labels_csv = os.path.join(
            self.data_dir, "four_findings_expert_labels_validation_labels.csv"
        )

        # Load all into one pool
        df_readers = pd.read_csv(readers_csv)
        df_test = pd.read_csv(test_labels_csv)
        df_val = pd.read_csv(val_labels_csv)
        expert_df = pd.concat([df_readers, df_test, df_val], ignore_index=True)

        # 2. Trigger selective download/extraction
        # needed_images = set(expert_df["Image ID"].unique())
        # self._download_and_extract_needed(needed_images)

        # 3. Process Labels
        concept_cols = ["Fracture", "Pneumothorax", "Airspace opacity", "Nodule/mass"]

        # if "Nodule or mass" in expert_df.columns:
        #     expert_df.rename(columns={"Nodule or mass": "Nodule/mass"}, inplace=True)

        for col in concept_cols:
            expert_df[col] = np.where(expert_df[col] == "YES", 1, 0)

        # Aggregate (Majority vote for multiple readers)
        df_agg = expert_df.groupby("Image ID")[concept_cols].mean().reset_index()
        for col in concept_cols:
            df_agg[col] = (df_agg[col] >= 0.5).astype(int)

        # Number of activated concepts
        concept_count = df_agg[concept_cols].sum(axis=1)

        # depending on the class
        if self.chx_multi_class:
            # TODO: maybe with the help of a clinitian we could rearrange something better
            df_agg["target"] = np.select(
                [
                    concept_count == 0, # 0 = Healthy
                    concept_count == 1, # 1 = Green
                    concept_count == 2, # 2 = Yellow
                    concept_count == 3, # 3 = Red
                    concept_count == 4, # 4 = Critical
                ],
                [0, 1, 2, 3, 4],
                default=-1
            )

            assert (df_agg["target"] >= 0).all(), "Invalid targets generated"
            class_names = ["Healthy", "Green code", "Yellow code", "Red code", "Critical code"]
            n_classes = 5
        else:
            df_agg["target"] = (concept_count == 0).astype(int)
            class_names = ["Abnormal", "Healthy"]
            n_classes = 2


        img_dir = os.path.join(self.data_dir, "images_nih", "images")
        df_agg["path"] = df_agg["Image ID"].apply(lambda x: os.path.join(img_dir, x))

        # Final check: only include what we successfully extracted
        df_agg = df_agg[df_agg["path"].apply(os.path.exists)].reset_index(drop=True)

        # 4. Split and Transform
        x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(
            df_agg["path"].values,
            df_agg[concept_cols].values,
            df_agg["target"].values,
            test_size=self.test_split,
            random_state=42,
        )
        x_train, x_val, c_train, c_val, y_train, y_val = train_test_split(
            x_train, c_train, y_train, test_size=self.val_split, random_state=42
        )

        # Label weights
        y_counts = np.bincount(y_train, minlength=n_classes)
        self.label_weights = torch.tensor(
            len(y_train) / (n_classes * y_counts),
            dtype=torch.float
        )

        # Concept weights
        for i in range(c_train.shape[1]):
            c_counts = np.bincount(c_train[:, i])
            if len(c_counts) < 2:
                weights = [1.0, 1.0]
            else:
                weights = len(c_train) / (2.0 * c_counts)
            self.concepts_weights.append(
                torch.tensor(weights, dtype=torch.float).to(self.device)
            )

        self.label_weights = self.label_weights.to(self.device)

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

        if not self.chx_multi_class:
            logic = HardLogic(
                lambda x: (np.sum(x, axis=1) == 0).astype(np.int64),
                n_concepts=1,
                concept_dim=4,
            )
        else:
            logic = HardLogic(
                lambda x: np.sum(x, axis=1).astype(np.int64),
                n_concepts=1,
                concept_dim=4,
            )

        return (
            CHXDataset(x_train, c_train, y_train, transform),
            CHXDataset(x_val, c_val, y_val, transform),
            CHXDataset(x_test, c_test, y_test, transform),
            (3, 224, 224),
            4,
            n_classes,
            1,
            class_names,
            concept_cols,
            logic,
            torch.nn.CrossEntropyLoss(weight=self.label_weights),
            self.concepts_weights,
            self.label_weights,
        )

def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for CHX
    chx_parser = subparsers.add_parser(
        "chx",
        help="Use CHX as dataset",
    )
    configure_global_arguments(chx_parser)

    subparsers = chx_parser.add_subparsers(dest="model")
    resnet18.configure_subparsers(subparsers)
    lenet.configure_subparsers(subparsers)
    linear.configure_subparsers(subparsers)
