from conformal.datasets.mnist import MNISTLoader
from conformal.datasets.mnistadd import MNISTAdditionDataset
from conformal.utils.logic import Logic
import torch


class MNISTSumParityDataset(MNISTAdditionDataset):
    def _label_aggregator(self, label1, label2):
        return (label1 + label2) % 2


class MNISTSumParityLoader(MNISTLoader):
    def load(self):
        base_train, base_val, base_test, _, _, _, _, _, _, _, _ = super().load()

        input_dim = (1, 28, 56)
        output_dim = 2
        concept_dim = 10
        n_images = 2

        class_names = [str(i) for i in range(output_dim)]
        concept_names = [str(i) for i in range(concept_dim)]

        logic = Logic(
            lambda x: (x[:, 0] + x[:, 1]) % 2,
            n_concepts=n_images, 
            concept_dim=concept_dim
        )

        return (
            MNISTSumParityDataset(base_train),
            MNISTSumParityDataset(base_val),
            MNISTSumParityDataset(base_test),
            input_dim,
            concept_dim,
            output_dim,
            n_images,
            class_names,
            concept_names,
            logic,
            torch.nn.NLLLoss(),
        )
