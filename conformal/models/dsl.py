import torch

from conformal.models.operators import (
    mnist_add_dsl_weights,
    mnist_sump_dsl_weights,
    boia_dsl_weights,
    chx_dsl_weights,
    derma_dsl_weights,
)
from conformal.models.nesy import NeSyModel
from conformal.utils.other import outer_product


def configure_global_arguments(parser):
    """Global arguments for DSL"""
    parser.add_argument(
        "--epsilon-symbols",
        type=float,
        help="Hyperparameter for learning symbols",
        default=0.2807344052335263,
    )
    parser.add_argument(
        "--epsilon-rules",
        type=float,
        help="Hyperparameter for learning rules",
        default=0.1077119516324264,
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

        self.weights = self._build_weights(
            n_images=n_images,
            concept_dim=concept_dim,
            output_dim=output_dim,
            dataset=dataset,
        )
        if dataset == "boia":
            for i in range(len(self.weights)):
                self.weights[i].requires_grad = True
        else:
            self.weights.requires_grad = True

        self.epsilon_symbols = epsilon_symbols
        self.epsilon_rules = epsilon_rules

        if dataset == "boia":
            self.arity = 1
        else:
            self.arity = self.weights.dim() - 1

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
            return boia_dsl_weights(
                n_images=n_images,
                concept_dim=concept_dim,
                output_dim=output_dim,
                device=self.device,
            )
        elif dataset == "chx":
            return chx_dsl_weights(
                n_images=n_images,
                concept_dim=concept_dim,
                output_dim=output_dim,
                device=self.device,
            )
        elif dataset == "derma":
            return derma_dsl_weights(
                n_images=n_images,
                concept_dim=concept_dim,
                output_dim=output_dim,
                device=self.device,
            )
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

    def get_rules_matrix(self, eval, index=None):
        """
        Returns:
        rules_weights: tensor of shape [C, ...] (arity dims)
        g_matrix:      tensor of same shape, symbolic outputs
        """
        # set weights
        if index is not None:
            weights = self.weights[index]
        else:
            weights = self.weights

        if eval:
            rules_weights, g_matrix = torch.max(
                torch.nn.functional.softmax(weights, dim=-1), dim=-1
            )
            return rules_weights, g_matrix

        # training (epsilon-greedy)
        rule_shape = weights.shape[:-1]  # e.g. (C,) or (C, C)
        n_outputs = weights.shape[-1]

        random_selection = (
            torch.rand(rule_shape, device=self.device) < self.epsilon_rules
        )

        symbol_index_random = torch.randint(n_outputs, rule_shape, device=self.device)

        _, symbol_index_max = torch.max(weights, dim=-1)

        g_matrix = torch.where(random_selection, symbol_index_random, symbol_index_max)

        rules_weights = torch.gather(
            torch.nn.functional.softmax(weights, dim=-1), -1, g_matrix.unsqueeze(-1)
        ).squeeze(-1)

        return rules_weights, g_matrix

    def inference(self, concepts, eval=False):
        """
        General DSL inference.
        """
        if self.dataset == "boia":
            return self._boia_inference(concepts, eval)

        truth_list = []
        symbol_list = []

        # If unary DSL
        if self.arity == 1:
            worlds = (
                outer_product(concepts.squeeze(1))
                if self.dataset in ["chx", "derma"]
                else outer_product(concepts)
            )
            tv, sym = self.epsilon_greedy(worlds, eval)
            truth_list.append(tv.squeeze())
            symbol_list.append(sym)
        else:
            # Multi-arity (e.g., MNIST add two digits)
            for i in range(self.arity):
                tv, sym = self.epsilon_greedy(concepts[:, i], eval)
                truth_list.append(tv.squeeze())
                symbol_list.append(sym)

        rules_weights, g_matrix = self.get_rules_matrix(eval)

        # Build index for rule lookup
        idx = tuple(symbol_list)
        rule_truth = rules_weights[idx]
        predicted_symbols = g_matrix[idx]

        # Combine rule truth with individual concept truth-values
        all_truths = torch.stack([rule_truth] + truth_list, dim=1)
        prediction_truth, _ = torch.min(all_truths, dim=1)

        output = (
            torch.nn.functional.one_hot(predicted_symbols, num_classes=self.output_dim)
            .float()
            .to(self.device)
        )

        return output, prediction_truth

    def _boia_inference(self, concepts, eval=False):
        concepts = concepts.squeeze(1)  # (B, num_concepts)

        # Forward
        fs_inputs = torch.stack(
            [
                concepts[:, 0],
                concepts[:, 1],
                concepts[:, 2],
                concepts[:, 3],
                concepts[:, 4],
                concepts[:, 10],
                concepts[:, 11],
                concepts[:, 12],
                concepts[:, 13],
            ],
            dim=1,
        )

        fs_worlds = outer_product(fs_inputs)
        f_tv, f_sym = self.epsilon_greedy(fs_worlds, eval)

        f_rules = self.weights[0]
        f_rules_w, f_g = (
            torch.max(torch.softmax(f_rules, dim=-1), dim=-1)
            if eval
            else self.get_rules_matrix(eval=False, index=0)
        )

        f_rule_truth = f_rules_w[f_sym]
        f_pred_sym = f_g[f_sym]

        f_truth = torch.min(
            torch.stack([f_rule_truth, f_tv.squeeze()], dim=1), dim=1
        ).values

        f_output = torch.nn.functional.one_hot(
            f_pred_sym, num_classes=f_rules.shape[-1]
        ).float()

        # Stop
        s_tv, s_sym = self.epsilon_greedy(fs_worlds, eval)

        s_rules = self.weights[1]
        s_rules_w, s_g = (
            torch.max(torch.softmax(s_rules, dim=-1), dim=-1)
            if eval
            else self.get_rules_matrix(eval=False, index=1)
        )

        s_rule_truth = s_rules_w[s_sym]
        s_pred_sym = s_g[s_sym]

        s_truth = torch.min(
            torch.stack([s_rule_truth, s_tv.squeeze()], dim=1), dim=1
        ).values

        s_output = torch.nn.functional.one_hot(
            s_pred_sym, num_classes=s_rules.shape[-1]
        ).float()

        # Left
        l_inputs = torch.stack(
            [
                concepts[:, 5],
                concepts[:, 6],
                concepts[:, 7],
                concepts[:, 8],
                concepts[:, 9],
                concepts[:, 14],
            ],
            dim=1,
        )

        l_worlds = outer_product(l_inputs)
        l_tv, l_sym = self.epsilon_greedy(l_worlds, eval)

        l_rules = self.weights[2]
        l_rules_w, l_g = (
            torch.max(torch.softmax(l_rules, dim=-1), dim=-1)
            if eval
            else self.get_rules_matrix(eval=False, index=2)
        )

        l_rule_truth = l_rules_w[l_sym]
        l_pred_sym = l_g[l_sym]

        l_truth = torch.min(
            torch.stack([l_rule_truth, l_tv.squeeze()], dim=1), dim=1
        ).values

        l_output = torch.nn.functional.one_hot(
            l_pred_sym, num_classes=l_rules.shape[-1]
        ).float()

        # Right
        r_inputs = torch.stack(
            [
                concepts[:, 15],
                concepts[:, 16],
                concepts[:, 17],
                concepts[:, 18],
                concepts[:, 19],
                concepts[:, 20],
            ],
            dim=1,
        )

        r_worlds = outer_product(r_inputs)
        r_tv, r_sym = self.epsilon_greedy(r_worlds, eval)

        r_rules = self.weights[3]
        r_rules_w, r_g = (
            torch.max(torch.softmax(r_rules, dim=-1), dim=-1)
            if eval
            else self.get_rules_matrix(eval=False, index=3)
        )

        r_rule_truth = r_rules_w[r_sym]
        r_pred_sym = r_g[r_sym]

        r_truth = torch.min(
            torch.stack([r_rule_truth, r_tv.squeeze()], dim=1), dim=1
        ).values

        r_output = torch.nn.functional.one_hot(
            r_pred_sym, num_classes=r_rules.shape[-1]
        ).float()

        pred = torch.cat([f_output, s_output, l_output, r_output], dim=1)
        truth = torch.cat(
            [
                f_truth.unsqueeze(1),
                s_truth.unsqueeze(1),
                l_truth.unsqueeze(1),
                r_truth.unsqueeze(1),
            ],
            dim=1,
        )

        return pred.to(self.device), truth.to(self.device)

    def compute_loss(
        self,
        dataset,
        criterion,
        conc_pred,
        concepts,
        output,
        target,
        label_weights,
        extra,
    ):
        """Return the DSL loss."""
        truth_values = extra  # fuzzy truth values from symbolic layer

        if dataset == "boia":
            total_loss = 0.0
            n_blocks = target.shape[1]

            for i in range(n_blocks):
                # Binary supervision for each block
                block_output = output[:, i]
                block_target = target[:, i]

                model_labels = (block_output.argmax(dim=-1) == block_target).float()
                truth_logits = torch.logit(truth_values[:, i], eps=1e-4)
                sample_weights = (
                    label_weights[i][block_target]
                    if label_weights is not None
                    else None
                )

                block_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    truth_logits, model_labels, weight=sample_weights, reduction="mean"
                )

                total_loss += block_loss

            return total_loss / n_blocks  # average over blocks

        model_labels = (output.argmax(dim=-1) == target).float()
        truth_logits = torch.logit(truth_values.view(-1), eps=1e-4)
        sample_weights = label_weights[target] if label_weights is not None else None

        return torch.nn.functional.binary_cross_entropy_with_logits(
            truth_logits, model_labels, weight=sample_weights, reduction="mean"
        )


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for DPL
    dpl_parser = subparsers.add_parser(
        "dsl",
        help="Use DSL as NeSy predictor",
    )
    configure_global_arguments(dpl_parser)
