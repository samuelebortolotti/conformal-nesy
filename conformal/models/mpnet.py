import torch
import torch.nn.functional as F

from conformal.models.base_lm import BaseLM
from conformal.models import dpl, ltn, linear_predictor, dsl


def configure_global_arguments(parser):
    pass


class MPNetSentence(BaseLM):

    def __init__(
        self,
        num_concept_dim=4,
        num_concepts=5,
        freeze_encoder=True,
    ):
        super().__init__(
            model_name="sentence-transformers/all-mpnet-base-v2",
            freeze_encoder=freeze_encoder,
            num_concept_dim=num_concept_dim,
            num_concepts=num_concepts,
        )

    def encode(self, input_ids, attention):
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention,
                )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention,
            )

        last_token = outputs.last_hidden_state[:, -1, :]
        return last_token


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for Bert
    mpnet_parser = subparsers.add_parser(
        "mpnet",
        help="Train an all-mpnet-base-v2 sentence model",
    )

    configure_global_arguments(mpnet_parser)

    subparsers = mpnet_parser.add_subparsers(dest="nesy")
    ltn.configure_subparsers(subparsers)
    dpl.configure_subparsers(subparsers)
    linear_predictor.configure_subparsers(subparsers)
    dsl.configure_subparsers(subparsers)
