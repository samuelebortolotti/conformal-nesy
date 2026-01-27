import torch

from conformal.models.operators import (
    mnist_add_dsl_weights,
    mnist_sump_dsl_weights,
    boia_dsl_weights,
    chx_dsl_weights,
    derma_dsl_weights,
)
from conformal.models.nesy import NeSyModel


def configure_global_arguments(parser):
    """Global arguments for DSL"""
    parser.add_argument(
        "--epsilon-symbols", type=float, help="Hyperparameter for learning symbols", default=0.2807344052335263
    )
    parser.add_argument(
        "--epsilon-rules", type=float, help="Hyperparameter for learning rules",  default=0.1077119516324264
    )


class DSL(NeSyModel):
    def __init__(
        self,
        n_images,
        encoder,
        entangled,
        concept_dim,
        output_dim,
        dataset,
        device,
        epsilon_symbols,
        epsilon_rules,
    ):
        super().__init__(
            n_images, encoder, entangled, concept_dim, output_dim, dataset, device
        )
        self.weights = self._build_weights(n_images, concept_dim, output_dim, dataset)
        self.weights.requires_grad = True
        self.epsilon_symbols = epsilon_symbols
        self.epsilon_rules = epsilon_rules

    def _build_weights(self, concept_dim, output_dim, n_images, dataset):
        """Build DSL weights"""
        if dataset == "mnistadd" or dataset == "mnisthalf":
            return mnist_add_dsl_weights(
                n_images=n_images,
                concept_dim=concept_dim,
                output_dim=output_dim,
                device=self.device,
            )
        elif dataset == "mnistsump":
            return mnist_sump_dsl_weights(
                n_images=n_images,
                concept_dim=concept_dim,
                output_dim=output_dim,
                device=self.device,
            )
        elif dataset == "boia":
            return boia_dsl_weights()
        elif dataset == "chx":
            return chx_dsl_weights()
        elif dataset == "derma":
            return derma_dsl_weights()
        raise NotImplementedError(
            f"DSL weights matrix for dataset {dataset} not implemented."
        )

    def epsilon_greedy(self, t, eval, dim=1):
        """Epsilon greedy strat for learning symbols"""
        if eval:
            truth_values, chosen_symbols = torch.max(t, dim=dim)
        else:
            random_selection = torch.rand((t.shape[0],)) < self.epsilon_symbols
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(t.shape[1], (t.shape[0],))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(t, dim=dim)

            chosen_symbols = torch.where(
                random_selection, symbol_index_random, symbol_index_max
            )
            truth_values = torch.gather(t, dim, chosen_symbols.view(-1, 1))

        return truth_values, chosen_symbols

    def get_rules_matrix(self, eval):
        """Get the rules matrix used for inference"""
        if eval:
            return torch.max(
                torch.nn.functional.softmax(self.weights, dim=2), dim=2, keepdim=True
            )
        else:
            n_digits = self.weights.shape[0]
            n_output_symbols = self.weights.shape[2]
            random_selection = torch.rand((n_digits, n_digits)) < self.epsilon_rules
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(n_output_symbols, (n_digits, n_digits))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(self.weights, dim=2)

            chosen_symbols = torch.where(
                random_selection, symbol_index_random, symbol_index_max
            )

            truth_values = torch.gather(
                torch.nn.functional.softmax(self.weights, dim=2),
                2,
                chosen_symbols.view(n_digits, n_digits, 1),
            ).view(n_digits, n_digits)

            return truth_values, chosen_symbols

    def inference(self, concepts):
        """DSL inference."""
        # TODO: only mnist for now
        truth_values_x, chosen_symbols_x = self.epsilon_greedy(concepts[:, 0], eval)
        truth_values_y, chosen_symbols_y = self.epsilon_greedy(concepts[:, 1], eval)
        rules_weights, g_matrix = self.get_rules_matrix(eval)

        symbols_truth_values = torch.concat(
            [
                rules_weights[chosen_symbols_x, chosen_symbols_y].view(-1, 1),
                truth_values_x.view(-1, 1),
                truth_values_y.view(-1, 1),
            ],
            dim=1,
        )
        predictions_truth_values, _ = torch.min(symbols_truth_values, 1)

        return g_matrix[chosen_symbols_x, chosen_symbols_y]


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for DPL
    dpl_parser = subparsers.add_parser(
        "dsl",
        help="Use DSL as NeSy predictor",
    )
    configure_global_arguments(dpl_parser)
