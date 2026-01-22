import torch.nn as nn
import torch

from conformal.utils.other import outer_product
from conformal.models.operators import mnist_circuit, mnist_sump_circuit


class DPL(nn.Module):
    def __init__(
        self,
        n_images,
        encoder,
        entangled,
        concept_dim,
        output_dim,
        dataset,
        device
    ):
        super().__init__()
        self.entangled = entangled
        self.encoder = encoder
        self.n_images = n_images
        self.concept_dim = concept_dim
        self.dataset = dataset
        self.device = device
        self.circuit = self._build_circuit(
            concept_dim, output_dim, n_images, dataset
        ).to(self.device)

    def _build_circuit(self, concept_dim, output_dim, n_images, dataset):
        if dataset == "mnistadd" or dataset == "mnisthalf":
            return mnist_circuit(sequence_len=n_images, n_digits=concept_dim, oputput_dim=output_dim)
        elif dataset == "mnistsump":
            return mnist_sump_circuit(sequence_len=n_images, n_digits=concept_dim, oputput_dim=output_dim)

        raise NotImplementedError(f"Circuit for dataset {dataset} not implemented.")

    def _dpl_inference(self, worlds):
        query_prob = torch.matmul(worlds, self.circuit)  # (B, nr_classes)
        return self._normalize(query_prob)

    def _normalize(self, x):
        eps = 1e-5
        x = x + eps
        with torch.no_grad():
            Z = torch.sum(x, dim=-1, keepdim=True)
        x = x / Z
        return x

    def get_concepts(self, x):
        if self.dataset in ["cub", "boia"]:
            return torch.nn.functional.sigmoid(self.encoder(x))

        return self._normalize(torch.nn.functional.softmax(self.encoder(x), dim=-1))

    def forward(self, x):
        if self.entangled:
            concepts = self.get_concepts(x)
            y = self._dpl_inference(concepts)

            return y, concepts
        else:
            
            xs = torch.chunk(x, self.n_images, dim=-1)
            concepts = [self.get_concepts(xi) for xi in xs]
            concepts = torch.stack(concepts, dim=1)

            # compue the possible words
            worlds = outer_product(concepts)
            y = self._dpl_inference(worlds)

            # keeping the concept semantic strict
            if self.dataset in ["cub", "boia"]:
                concepts_1 = concepts.transpose(1, 2)
                concepts_0 = 1 - concepts_1
                concepts = torch.cat(
                    [concepts_0, concepts_1], dim=2
                )
                concepts = self._normalize(concepts)

            if self.dataset in ["boia"]:
                y = torch.stack([1 - y, y], dim=2)
                y = self._normalize(y)

            return y, concepts
