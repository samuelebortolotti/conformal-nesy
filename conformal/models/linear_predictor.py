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

        in_dim = concept_dim**n_images
        if self.dataset == "boia":
            in_dim = concept_dim

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_dim, output_dim), torch.nn.Softmax(dim=-1)
        ).to(self.device)

        self._init_linear()

    def _init_linear(self):
        """Initialize the linear layer"""
        torch.nn.init.normal_(
            self.linear[0].weight,
            0,
            5,
        )

    def inference(self, concepts):
        """Linear predictor inference."""
        if self.dataset == "boia":
            out = self.linear(concepts)
        else:
            worlds = (
                outer_product(concepts.squeeze(1))
                if self.dataset in ["chx", "derma"]
                else outer_product(concepts)
            )
            out = self.linear(worlds)
        return self._normalize(out)


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for Linear
    linear_parser = subparsers.add_parser(
        "linpred",
        help="Use Linear Predictor as NeSy predictor",
    )
    configure_global_arguments(linear_parser)
