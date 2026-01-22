import torch.nn as nn
import torch
from conformal.models.operators import mnist_logic


class LTN(nn.Module):
    def __init__(
        self, n_images, encoder, entangled, concept_dim, output_dim, dataset, device
    ):
        super().__init__()
        self.entangled = entangled
        self.encoder = encoder
        self.n_images = n_images
        self.concept_dim = concept_dim
        self.dataset = dataset
        self.device = device
        self.output_dim = output_dim
        self.hard_logic = self._build_hard_logic(
            concept_dim, output_dim, n_images, dataset, entangled
        )

    def _build_hard_logic(self, concept_dim, output_dim, n_images, dataset, entangled):
        if dataset == "mnistadd" or dataset == "mnisthalf":
            return mnist_logic(n_images, entangled)

        raise NotImplementedError(f"Circuit for dataset {dataset} not implemented.")

    def _ltn_inference(self, concepts):
        concepts = torch.argmax(concepts, dim=-1)
        y = self.hard_logic(concepts)
        print(y.shape, y, self.output_dim)
        query_prob = torch.nn.functional.one_hot(y, num_classes=self.output_dim).float()
        return query_prob

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
            y = self._ltn_inference(concepts)

            return y, concepts
        else:

            xs = torch.chunk(x, self.n_images, dim=-1)
            concepts = [self.get_concepts(xi) for xi in xs]
            concepts = torch.stack(concepts, dim=1)

            y = self._ltn_inference(concepts)

            # keeping the concept semantic strict
            if self.dataset in ["cub", "boia"]:
                concepts_1 = concepts.transpose(1, 2)
                concepts_0 = 1 - concepts_1
                concepts = torch.cat([concepts_0, concepts_1], dim=2)
                concepts = self._normalize(concepts)

            if self.dataset in ["boia"]:
                y = torch.stack([1 - y, y], dim=2)
                y = self._normalize(y)

            return y, concepts


# MISSING CRITERION
