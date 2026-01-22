import os
import torch
from torch.utils.data import Dataset, random_split
import pickle

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

    def __init__(self, root="./data/bdd2048", val_split=0.1):
        self.root = root
        self.val_split = val_split
        self.transform = None

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
            self._label_aggregator,
            torch.nn.NLLLoss(),
        )
