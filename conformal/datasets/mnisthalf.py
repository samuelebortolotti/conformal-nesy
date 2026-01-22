from conformal.datasets.mnist import MNISTLoader
from conformal.datasets.mnistadd import MNISTAdditionDataset
from conformal.utils.logic import Logic
import torch


class MNISTHalfDataset(MNISTAdditionDataset):
    def _label_aggregator(self, label1, label2):
        return label1 + label2


class MNISTHalfLoader(MNISTLoader):
    def load(self):
        def _in_distribution_filter(label1, label2):
            allowed_pairs = {(0, 0), (0, 1), (2, 3), (2, 4)}
            return (label1, label2) in allowed_pairs

        def _out_of_distribution_filter(label1, label2):
            return label1 < 5 and label2 < 5

        base_train, base_val, base_test, _, _, _, _, _, _, _, _ = super().load()

        input_dim = (1, 28, 56)
        output_dim = 9
        concept_dim = 5
        n_images = 2

        class_names = [str(i) for i in range(output_dim)]
        concept_names = [str(i) for i in range(concept_dim)]

        logic = Logic(
            lambda x: x[:, 0] + x[:, 1],
            n_concepts=n_images, 
            concept_dim=concept_dim
        )

        return (
            MNISTHalfDataset(
                base_train, 
                filter_fn=_in_distribution_filter
            ),
            MNISTHalfDataset(
                base_val, 
                filter_fn=_in_distribution_filter
            ),  # if not self.active else MNISTHalfDataset(base_val, filter_fn=_out_of_distribution_filter),
            MNISTHalfDataset(
                base_test, 
                filter_fn=_in_distribution_filter #_out_of_distribution_filter),
            ),
            input_dim,
            concept_dim,
            output_dim,
            n_images,
            class_names,
            concept_names,
            logic,
            torch.nn.NLLLoss(),
        )
