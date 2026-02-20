from conformal.models.base_lm import BaseLM
from conformal.models import dpl, ltn, linear_predictor, dsl


def configure_global_arguments(parser):
    pass


class Lama(BaseLM):
    def __init__(
        self,
        num_concept_dim=4,
        num_concepts=5,
        freeze_encoder=True,
    ):
        super().__init__(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            freeze_encoder=freeze_encoder,
            num_concept_dim=num_concept_dim,
            num_concepts=num_concepts,
        )


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for lama
    lama_parser = subparsers.add_parser(
        "lama",
        help="Train a lama model",
    )

    configure_global_arguments(lama_parser)

    subparsers = lama_parser.add_subparsers(dest="nesy")
    ltn.configure_subparsers(subparsers)
    dpl.configure_subparsers(subparsers)
    linear_predictor.configure_subparsers(subparsers)
    dsl.configure_subparsers(subparsers)
