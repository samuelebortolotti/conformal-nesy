import torch
import torch.nn as nn


class NeSyModel(nn.Module):
    def __init__(
        self, n_images, encoder, entangled, concept_dim, output_dim, dataset, device
    ):
        super().__init__()
        self.n_images = n_images
        self.encoder = encoder
        self.entangled = entangled
        self.concept_dim = concept_dim
        self.output_dim = output_dim
        self.dataset = dataset
        self.device = device

    def _normalize(self, x):
        """Shared normalization to ensure valid probability distributions."""
        eps = 1e-5
        x = x + eps
        with torch.no_grad():
            Z = torch.sum(x, dim=-1, keepdim=True)
        return x / Z

    def get_concepts(self, x):
        """Common logic for extracting concepts from the encoder."""
        if self.dataset in ["boia", "chx", "derma"]:
            # Independent binary concepts
            c = torch.sigmoid(self.encoder(x))
            c = torch.stack([1 - c, c], dim=-1)
        else:
            # Categorical concepts (e.g., MNIST digits)
            c = torch.softmax(self.encoder(x), dim=-1)

        return self._normalize(c)

    def forward(self, x):
        if self.entangled:
            concepts = self.get_concepts(x)
            y, extra = self.inference(concepts)

            return y, concepts, extra
        else:

            xs = torch.chunk(x, self.n_images, dim=-1)
            concepts = [self.get_concepts(xi) for xi in xs]
            concepts = torch.stack(concepts, dim=1)

            # inference
            y, extra = self.inference(concepts)

            # Special case for BDD-OIA
            if self.dataset in ["boia"]:
                y = y.view(y.size(0), -1, 2)

            return y, concepts, extra

    def inference(self, concepts):
        """Abstract method to be overridden by LTN, DPL and Linear Predictor."""
        raise NotImplementedError("Subclasses must implement the inference method.")

    def compute_loss(self, dataset, criterion, conc_pred, concepts, output, target, label_weights, extra):
        """DPL and Linear Predictor standard loss on labels. LTN overrides it"""
        if isinstance(criterion, torch.nn.NLLLoss):
            output = output.log()

        if dataset != "boia":
            loss = criterion(output, target)
        else:
            loss = 0.0
            for i in range(output.size(1)):
                sample_weights = (
                    label_weights[i] if label_weights is not None else None
                )
                loss = torch.nn.functional.nll_loss(
                    output.permute(0, 2, 1), target, weight=sample_weights
                )
            loss /= output.size(1)

        return loss
