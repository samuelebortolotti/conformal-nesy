import torch

from conformal.utils.other import outer_product
from conformal.models.nesy import NeSyModel


def configure_global_arguments(parser):
    """Global arguments for Linear Predictor"""
    pass


class LinearPredictor(NeSyModel):
    def __init__(
        self, n_images, encoder, entangled, concept_dim, output_dim, dataset, device
    ):
        super().__init__(
            n_images, encoder, entangled, concept_dim, output_dim, dataset, device
        )

        activation = torch.nn.Softmax(dim=-1)

        if self.dataset == "boia":
            in_dim = concept_dim * 2
            activation = torch.nn.Sigmoid()
        elif "mnist" in self.dataset:
            in_dim = concept_dim**n_images
        else:
            in_dim = 2**concept_dim

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_dim, output_dim), activation
        ).to(self.device)


    def _inference_boia(self, concepts):
        out = self.linear(concepts.view(concepts.shape[0], -1))
        out = torch.stack([out, 1.0 - out], dim=-1).reshape(out.shape[0], -1)
        return out

    def inference(self, concepts):
        """Linear predictor inference."""
        log_odds = torch.log(concepts)

        if self.dataset == "boia":
            out = self._inference_boia(log_odds.squeeze(1))
        else:
            worlds = (
                outer_product(log_odds.squeeze(1))
                if self.dataset in ["chx", "derma"]
                else outer_product(log_odds)
            )
            out = self.linear(worlds)
        return self._normalize(out), None


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for Linear
    linear_parser = subparsers.add_parser(
        "linpred",
        help="Use Linear Predictor as NeSy predictor",
    )
    configure_global_arguments(linear_parser)
