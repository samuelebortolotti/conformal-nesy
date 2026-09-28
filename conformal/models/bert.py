from conformal.models.base_lm import BaseLM
from conformal.models import dpl, ltn, linear_predictor, dsl


def configure_global_arguments(parser):
    pass


class Bert(BaseLM):
    def __init__(
        self,
        num_concept_dim=4,
        num_concepts=5,
        freeze_encoder=True,
    ):
        super().__init__(
            model_name="bert-base-uncased",
            freeze_encoder=freeze_encoder,
            num_concept_dim=num_concept_dim,
            num_concepts=num_concepts,
        )


def configure_subparsers(subparsers):
    bert_parser = subparsers.add_parser(
        "bert",
        help="Train a bert model",
    )

    configure_global_arguments(bert_parser)

    subparsers = bert_parser.add_subparsers(dest="nesy")
    ltn.configure_subparsers(subparsers)
    dpl.configure_subparsers(subparsers)
    linear_predictor.configure_subparsers(subparsers)
    dsl.configure_subparsers(subparsers)
