import os
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from conformal.utils.logic import HardLogic, FactorizedBoiaLogic
from conformal.general_utils import log
from conformal.models import resnet18, lenet, linear
from conformal.models import clip_encoder
import pickle


def configure_global_arguments(parser):
    """Global arguments for BOIA"""
    parser.add_argument(
        "--boia-raw-root",
        default=None,
        help="Path to raw BDD-OIA image frames (required when using --model clip).",
    )


CONCEPTS_ORDER = {
    "red_light": 3,
    "green_light": 0,
    "car": 5,
    "person": 6,
    "rider": 7,
    "other_obstacle": 8,
    "follow": 1,
    "stop_sign": 4,
    "left_lane": 18,
    "left_green_light": 19,
    "left_follow": 20,
    "no_left_lane": 9,
    "left_obstacle": 10,
    "left_solid_line": 11,
    "right_lane": 12,
    "right_green_light": 13,
    "right_follow": 14,
    "no_right_lane": 15,
    "right_obstacle": 16,
    "right_solid_line": 17,
    "clear": 2,
}


class BOIADataset(Dataset):
    """
    Torch Dataset for BOIA (BDD) dataset.
    Returns: (image_tensor, class_label[:4], concept_vector)
    """

    def __init__(
        self,
        pkl_file_path,
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=None,
        n_class_attr=2,
        transform=None,
        c_sup=1,
        which_c=[-1],
    ):
        """
        Args:
            pkl_file_path: path to pickle file
            use_attr: whether to load concepts/attributes
            no_img: if True, return dummy image tensors
            uncertain_label: use uncertain attribute labels
            image_dir: folder containing preprocessed images (.pt)
            n_class_attr: number of classes per attribute
            transform: image transform (optional)
            c_sup: concept supervision threshold
            which_c: list of concept indices to keep (-1 = all)
        """
        self.data = pickle.load(open(pkl_file_path, "rb"))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

        self.c_sup = c_sup
        self.which_c = which_c

        self.is_train = "train" in pkl_file_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data["img_path"]

        # Preprocessed paths
        t_path = img_path[:-4] + ".pt"
        img_path_full = os.path.join(self.image_dir, "inputs", t_path)
        lab_path = os.path.join(self.image_dir, "labels", t_path)
        con_path = os.path.join(self.image_dir, "concepts", t_path)

        # Load tensors
        img = torch.load(img_path_full).squeeze(0)
        class_label = torch.load(lab_path).squeeze(0)[:4]  # only first 4 classes
        attr_label = torch.load(con_path).squeeze(0)

        # Filter concepts according to supervision
        if self.c_sup != 1:
            if self.r_seq[idx] > self.c_sup:
                attr_label[:] = -1
            elif not (self.which_c[0] == -1):
                for c in range(attr_label.shape[0]):
                    if c not in self.which_c:
                        attr_label[c] = -1
            else:
                for k, order in CONCEPTS_ORDER.items():
                    if k not in [
                        "red_light",
                        "green_light",
                        "car",
                        "person",
                        "rider",
                        "other_obstacle",
                        "stop_sign",
                        "right_green_light",
                        "left_green_light",
                    ]:
                        attr_label[order] = -1

        # Apply transform if defined
        if self.transform and not self.no_img:
            img = self.transform(img)

        if self.no_img:
            img = torch.zeros((3, 224, 224), dtype=torch.float32)

        return img, attr_label, class_label.to(torch.long)


class BOIALoader:
    """
    Simplified BOIA loader following MNIST/CUB style.
    Returns:
    train, val, test, input_dim, concept_dim, output_dim, n_images, class_names, concept_names, label_aggregator
    """

    def __init__(self, root="./data/bdd2048", val_split=0.1, device="cuda"):
        self.root = root
        self.val_split = val_split
        self.transform = None
        self.label_weights = []
        self.concept_weights = []
        self.device = device

    def _label_aggregator(self, label):
        # Identity function (can be customized)
        return label

    def load(self):
        """Load BOIA datasets and return metadata."""

        train_path = os.path.join(self.root, "train_BDD_OIA.pkl")
        val_path = os.path.join(self.root, "val_BDD_OIA.pkl")
        test_path = os.path.join(self.root, "test_BDD_OIA.pkl")
        image_dir = self.root

        # Load datasets
        full_train = BOIADataset(
            pkl_file_path=train_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=os.path.join(image_dir, "train"),
            n_class_attr=2,
            transform=self.transform,
        )
        val_dataset = BOIADataset(
            pkl_file_path=val_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=os.path.join(image_dir, "val"),
            n_class_attr=2,
            transform=self.transform,
        )
        test_dataset = BOIADataset(
            pkl_file_path=test_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=os.path.join(image_dir, "test"),
            n_class_attr=2,
            transform=self.transform,
        )

        val_size = int(self.val_split * len(full_train))
        train_size = len(full_train) - val_size
        train_dataset, _ = random_split(full_train, [train_size, val_size])

        # Metadata
        input_dim = 2048
        concept_dim = len(CONCEPTS_ORDER)
        output_dim = 4
        n_images = 1
        class_names = [f"Class_{i}" for i in range(output_dim)]
        concept_names = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)

        # Fuzzy intepretation that is still hard and ok
        # OR (A∨B): Represented as torch.clamp(A + B, 0, 1).
        # AND (A∧B): Represented as A * B.
        # NOT (¬A): Represented as 1 - A.

        _boia_lambda = lambda x: np.stack(
            [
                # 1. STOP
                # red_light + stop_sign + obstacle
                np.clip(
                    x[:, 3]
                    + x[:, 4]
                    + np.clip(x[:, 5] + x[:, 6] + x[:, 7] + x[:, 8], 0, 1),
                    0,
                    1,
                ),
                # 2. MOVE_FORWARD
                # (green_light + follow + road_clear) * (1 - stop)
                np.clip(
                    np.clip(
                        x[:, 0]
                        + x[:, 1]
                        + (
                            1 - np.clip(x[:, 5] + x[:, 6] + x[:, 7] + x[:, 8], 0, 1)
                        ),
                        0,
                        1,
                    )
                    * (
                        1
                        - np.clip(
                            x[:, 3]
                            + x[:, 4]
                            + np.clip(x[:, 5] + x[:, 6] + x[:, 7] + x[:, 8], 0, 1),
                            0,
                            1,
                        )
                    ),
                    0,
                    1,
                ),
                # 3. TURN_LEFT
                # (can_turn_left) * (1 - cannot_turn_left)
                np.clip(
                    np.clip(x[:, 18] + x[:, 19] + x[:, 20], 0, 1)
                    * (1 - np.clip(x[:, 9] + x[:, 10] + x[:, 11], 0, 1)),
                    0,
                    1,
                ),
                # 4. TURN_RIGHT
                # (can_turn_right) * (1 - cannot_turn_right)
                np.clip(
                    np.clip(x[:, 12] + x[:, 13] + x[:, 14], 0, 1)
                    * (1 - np.clip(x[:, 15] + x[:, 16] + x[:, 17], 0, 1)),
                    0,
                    1,
                ),
            ],
            axis=1,
        )
        logic = FactorizedBoiaLogic(
            _boia_lambda,
            n_concepts=n_images,
            concept_dim=concept_dim,
        )

        log("Calculating class and concept weights for BOIA...", "INFO")
        all_concepts = []
        all_labels = []

        for i in range(len(full_train)):
            img_data = full_train.data[i]
            t_path = img_data["img_path"][:-4] + ".pt"

            lab_path = os.path.join(image_dir, "train", "labels", t_path)
            con_path = os.path.join(image_dir, "train", "concepts", t_path)

            all_labels.append(torch.load(lab_path).squeeze(0)[:4].numpy())
            all_concepts.append(torch.load(con_path).squeeze(0).numpy())

        all_labels = np.array(all_labels)
        all_concepts = np.array(all_concepts)

        for i in range(all_labels.shape[1]):
            counts = np.bincount(all_labels[:, i].astype(int), minlength=2)
            w = len(all_labels) / (2.0 * counts)
            self.label_weights.append(
                torch.tensor(w, dtype=torch.float32).to(self.device)
            )

        for i in range(all_concepts.shape[1]):
            counts = np.bincount(all_concepts[:, i].astype(int), minlength=2)
            if counts[1] == 0 or counts[0] == 0:  # Handle rare concepts
                w = [1.0, 1.0]
            else:
                w = len(all_concepts) / (2.0 * counts)
            self.concept_weights.append(
                torch.tensor(w, dtype=torch.float32).to(self.device)
            )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            input_dim,
            concept_dim,
            output_dim,
            n_images,
            class_names,
            concept_names,
            logic,
            torch.nn.NLLLoss(),
            self.concept_weights,
            self.label_weights,
        )


class BOIARawDataset(Dataset):
    """BOIA dataset that loads raw JPEG frames instead of precomputed .pt feature tensors."""

    def __init__(self, pkl_file_path, raw_image_dir, precomputed_dir, transform=None):
        from PIL import Image as PILImage
        self._PIL = PILImage
        self.data = pickle.load(open(pkl_file_path, "rb"))
        self.raw_image_dir = raw_image_dir
        self.precomputed_dir = precomputed_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data["img_path"]
        t_path = img_path[:-4] + ".pt"

        raw_path = os.path.join(self.raw_image_dir, img_path)
        img = self._PIL.open(raw_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        lab_path = os.path.join(self.precomputed_dir, "labels", t_path)
        con_path = os.path.join(self.precomputed_dir, "concepts", t_path)
        class_label = torch.load(lab_path).squeeze(0)[:4]
        attr_label = torch.load(con_path).squeeze(0)

        return img, attr_label, class_label.to(torch.long)


class BOIARawImageLoader(BOIALoader):
    """BOIA loader that uses raw video frames for CLIP-compatible image input.

    Requires raw BDD-OIA JPEG frames in raw_root/{train,val,test}/img_path.
    Labels and concepts are still read from precomputed .pt files in root/.
    """

    def __init__(self, raw_root=None, root="./data/bdd2048", val_split=0.1, device="cuda"):
        super().__init__(root=root, val_split=val_split, device=device)
        self.raw_root = raw_root or root

    def load(self):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        train_path = os.path.join(self.root, "train_BDD_OIA.pkl")
        val_path = os.path.join(self.root, "val_BDD_OIA.pkl")
        test_path = os.path.join(self.root, "test_BDD_OIA.pkl")
        image_dir = self.root

        full_train = BOIARawDataset(
            pkl_file_path=train_path,
            raw_image_dir=os.path.join(self.raw_root, "train"),
            precomputed_dir=os.path.join(image_dir, "train"),
            transform=transform,
        )
        val_dataset = BOIARawDataset(
            pkl_file_path=val_path,
            raw_image_dir=os.path.join(self.raw_root, "val"),
            precomputed_dir=os.path.join(image_dir, "val"),
            transform=transform,
        )
        test_dataset = BOIARawDataset(
            pkl_file_path=test_path,
            raw_image_dir=os.path.join(self.raw_root, "test"),
            precomputed_dir=os.path.join(image_dir, "test"),
            transform=transform,
        )

        val_size = int(self.val_split * len(full_train))
        train_size = len(full_train) - val_size
        train_dataset, _ = random_split(full_train, [train_size, val_size])

        input_dim = (3, 224, 224)
        concept_dim = len(CONCEPTS_ORDER)
        output_dim = 4
        n_images = 1
        class_names = [f"Class_{i}" for i in range(output_dim)]
        concept_names = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)

        _boia_lambda_raw = lambda x: np.stack(
            [
                np.clip(x[:, 3] + x[:, 4] + np.clip(x[:, 5] + x[:, 6] + x[:, 7] + x[:, 8], 0, 1), 0, 1),
                np.clip(np.clip(x[:, 0] + x[:, 1] + (1 - np.clip(x[:, 5] + x[:, 6] + x[:, 7] + x[:, 8], 0, 1)), 0, 1) * (1 - np.clip(x[:, 3] + x[:, 4] + np.clip(x[:, 5] + x[:, 6] + x[:, 7] + x[:, 8], 0, 1), 0, 1)), 0, 1),
                np.clip(np.clip(x[:, 18] + x[:, 19] + x[:, 20], 0, 1) * (1 - np.clip(x[:, 9] + x[:, 10] + x[:, 11], 0, 1)), 0, 1),
                np.clip(np.clip(x[:, 12] + x[:, 13] + x[:, 14], 0, 1) * (1 - np.clip(x[:, 15] + x[:, 16] + x[:, 17], 0, 1)), 0, 1),
            ],
            axis=1,
        )
        logic = FactorizedBoiaLogic(
            _boia_lambda_raw,
            n_concepts=n_images,
            concept_dim=concept_dim,
        )

        log("Calculating class and concept weights for BOIA (raw images)...", "INFO")
        all_concepts = []
        all_labels = []

        for i in range(len(full_train)):
            img_data = full_train.dataset.data[i]
            t_path = img_data["img_path"][:-4] + ".pt"
            lab_path = os.path.join(image_dir, "train", "labels", t_path)
            con_path = os.path.join(image_dir, "train", "concepts", t_path)
            all_labels.append(torch.load(lab_path).squeeze(0)[:4].numpy())
            all_concepts.append(torch.load(con_path).squeeze(0).numpy())

        all_labels = np.array(all_labels)
        all_concepts = np.array(all_concepts)

        for i in range(all_labels.shape[1]):
            counts = np.bincount(all_labels[:, i].astype(int), minlength=2)
            w = len(all_labels) / (2.0 * counts)
            self.label_weights.append(torch.tensor(w, dtype=torch.float32).to(self.device))

        for i in range(all_concepts.shape[1]):
            counts = np.bincount(all_concepts[:, i].astype(int), minlength=2)
            w = [1.0, 1.0] if counts[1] == 0 or counts[0] == 0 else len(all_concepts) / (2.0 * counts)
            self.concept_weights.append(torch.tensor(w, dtype=torch.float32).to(self.device))

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            input_dim,
            concept_dim,
            output_dim,
            n_images,
            class_names,
            concept_names,
            logic,
            torch.nn.NLLLoss(),
            self.concept_weights,
            self.label_weights,
        )


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for BOIA
    boia_parser = subparsers.add_parser(
        "boia",
        help="Use BOIA as dataset",
    )
    configure_global_arguments(boia_parser)

    subparsers = boia_parser.add_subparsers(dest="model")
    resnet18.configure_subparsers(subparsers)
    lenet.configure_subparsers(subparsers)
    linear.configure_subparsers(subparsers)
    clip_encoder.configure_subparsers(subparsers)
