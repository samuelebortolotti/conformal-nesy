import torch
import numpy as np
import itertools
from conformal.general_utils import log


class ConformalPredictor:
    def __init__(
        self,
        model,
        device,
        logic,
        concept_dim=10,
        n_concepts=2,
        multiconcepts=False,
        multilabel=False,
        bonferroni=False,
    ):
        """
        model: PyTorch model returning (label_pred, concept_pred)
        device: 'cuda' or 'cpu'
        logic: object that converts concept predictions to label probabilities
        """
        self.model = model
        self.device = device
        self.logic = logic
        self.concept_dim = concept_dim
        self.n_concepts = n_concepts
        self.multiconcepts = multiconcepts
        self.multilabel = multilabel

        self.per_concept_thresholds = None
        self.label_threshold = None
        self.bonferroni = bonferroni

    @torch.no_grad()
    def compute_conformity_scores(self, dl):
        """
        Compute per-concept conformity scores on a calibration dataset.
        Conformity score = 1 - predicted probability of the true concept.

        Returns:
            scores: numpy array of shape (N_samples, N_concepts)
        """
        self.model.eval()
        all_scores = []

        for data, concepts, _ in dl:
            data, concepts = data.to(self.device), concepts.to(self.device)
            _, conc_pred = self.model(data)

            # For each concept, compute 1 - probability of true label
            if self.multiconcepts:
                batch_scores = torch.stack(
                    [
                        1
                        - conc_pred[:, i, j, :][
                            range(concepts.size(0)), concepts[:, j].long()
                        ]
                        for i in range(self.n_concepts)
                        for j in range(concepts.size(1))
                    ],
                    dim=1,
                )
            else:
                batch_scores = torch.stack(
                    [
                        1
                        - conc_pred[:, i, :][
                            range(concepts.size(0)), concepts[:, i].long()
                        ]
                        for i in range(concepts.size(1))
                    ],
                    dim=1,
                )

            all_scores.append(batch_scores.cpu().numpy())

        return np.concatenate(all_scores, axis=0)

    @torch.no_grad()
    def compute_label_scores(self, dl):
        """
        Compute conformity scores for final labels based on concept predictions.
        Conformity score = 1 - predicted probability of the true label.
        """
        self.model.eval()
        all_scores = []

        for data, _, labels in dl:
            data, labels = data.to(self.device), labels.to(self.device)
            label_pred, _ = self.model(data)

            if self.multilabel:
                batch_scores = torch.stack(
                    [
                        1
                        - label_pred[:, i, :][
                            range(labels.size(0)), labels[:, i].long()
                        ]
                        for i in range(labels.size(1))
                    ],
                    dim=1,
                )
            else:
                batch_scores = 1 - label_pred[range(len(labels)), labels.long()]
            all_scores.append(batch_scores.cpu().numpy())

        return np.concatenate(all_scores)

    def calibrate_per_concept(self, dl, alpha=0.1):
        """
        Calibrate thresholds individually per concept using quantiles.
        """
        scores = self.compute_conformity_scores(dl)

        # Determine the effective alpha per concept
        k = scores.shape[1]  # Number of concepts
        eff_alpha = alpha / k if self.bonferroni else alpha

        if self.bonferroni:
            log(
                f"[Conformal] Applying Bonferroni: Joint alpha {alpha} -> Per-concept alpha {eff_alpha:.4f}",
                "INFO"
            )

        # Compute the (1 - eff_alpha) quantile for each concept column
        self.per_concept_thresholds = np.quantile(scores, 1 - eff_alpha, axis=0)
        log(f"[Conformal] Per-concept thresholds: {self.per_concept_thresholds}", "INFO")

    def calibrate_labels(self, dl, alpha=0.1):
        """
        Calibrate threshold for the final label set.
        """
        scores = self.compute_label_scores(dl)

        # Determine the effective alpha per label
        # Number of labels
        k = 1
        if len(scores.shape) > 1: 
            k = scores.shape[1] 
        eff_alpha = alpha / k if self.bonferroni else alpha

        if self.bonferroni:
            log(
                f"[Conformal] Applying Bonferroni (Multilabel): Joint alpha {alpha} -> Per-label alpha {eff_alpha:.4f}",
                "INFO"
            )

        self.label_threshold = np.quantile(scores, 1 - eff_alpha, axis=0)
        log(f"[Conformal] Label threshold: {self.label_threshold}", "INFO")


    def _build_concept_sets_for_batch(self, conc_pred):
        """
        Private helper: Converts raw concept predictions (tensor)
        into a list of included label sets using thresholds.
        """
        batch_sets = []

        # conc_pred shape: (Batch, N_Concepts, N_Classes)
        for i in range(conc_pred.size(0)):  # Per sample
            sample_set = []
            for j in range(conc_pred.size(1)):  # Per concept

                # Include class k if: 1 - prob[k] <= threshold
                if self.multiconcepts:
                    included_list = []
                    for k in range(self.per_concept_thresholds.shape[0]):
                        included = (
                            torch.where(
                                1 - conc_pred[i, j, k, :][:]
                                <= self.per_concept_thresholds[k]
                            )[0]
                            .cpu()
                            .numpy()
                        )
                        included_list.append(included)
                    # NOTE: otherwise it becomes a list of another one
                    batch_sets.append(included_list)
                else:
                    included = (
                        torch.where(
                            1 - conc_pred[i, j, :] <= self.per_concept_thresholds[j]
                        )[0]
                        .cpu()
                        .numpy()
                    )
                    sample_set.append(included)
                    batch_sets.append(sample_set)
        return batch_sets


    def _generate_combinations(self, marginal_list):
        """Helper to process empty sets and generate Cartesian products."""
        processed = [
            s if s.size > 0 else np.arange(self.concept_dim) 
            for s in marginal_list
        ]
        return np.array(list(itertools.product(*processed)))


    @torch.no_grad()
    def predict_concepts(self, dl):
        """
        Predict conformal sets per concept using per-concept thresholds.
        Returns:
            prediction_sets[sample][concept] = array of included labels
        """
        if self.per_concept_thresholds is None:
            raise ValueError("Run calibrate_per_concept first.")

        self.model.eval()
        all_tuple_sets = []

        for data, _, _ in dl:
            data = data.to(self.device)
            _, conc_pred = self.model(data)

            # TODO: unless the concept outcome is entangled
            # Get marginals first
            batch_marginal_sets = self._build_concept_sets_for_batch(conc_pred)
            
            # Convert to tuples immediately
            for sample_marginal in batch_marginal_sets:
                tuples = self._generate_combinations(sample_marginal)
                all_tuple_sets.append(tuples)

        return all_tuple_sets


    def _refine_concept_prediction_set(
        self, label_prediction_set, concept_prediction_tuples
    ):
        """
        Refines concept tuples based on label predictions.
        Returns only the filtered tuples.
        """
        refined_batch_tuples = []

        for labels, tuples in zip(label_prediction_set, concept_prediction_tuples):
            if tuples.size == 0 or len(labels) == 0:
                refined_batch_tuples.append(tuples)
                continue

            derived_labels = self.logic.forward(tuples)
            if hasattr(derived_labels, "cpu"):
                derived_labels = derived_labels.cpu().numpy()
            
            # Keep tuples that result in an allowed label
            mask = np.isin(derived_labels, labels)
            valid_tuples = tuples[mask]
            refined_batch_tuples.append(valid_tuples)

        return refined_batch_tuples


    def _compute_derived_labels_from_tuples(self, batch_concept_tuples):
        """
        Derives labels from concept tuples using Hard Logic.
        
        Args:
            batch_concept_tuples: List of np.ndarrays, each of shape (N_combinations, N_concepts)
        Returns:
            List of np.ndarrays containing unique predicted labels per sample.
        """
        if not batch_concept_tuples:
            return []

        # Track indices to split the massive batch later
        sample_counts = [t.shape[0] for t in batch_concept_tuples]

        # Flatten all tuples into one massive matrix for a single logic pass
        big_matrix = np.concatenate(batch_concept_tuples, axis=0)
        # Vectorized Logic Pass
        raw_labels = self.logic.forward(big_matrix)

        if hasattr(raw_labels, "cpu"):
            raw_labels = raw_labels.cpu().numpy()
        
        # Ensure labels are flat (for single-label classification tasks)
        raw_labels = np.array(raw_labels).ravel()

        # Reconstruct per-sample unique label sets
        batch_label_sets = []
        cursor = 0
        for count in sample_counts:
            # Slice labels belonging to this specific sample
            sample_preds = raw_labels[cursor : cursor + count]
            unique_labels = np.unique(sample_preds)
            batch_label_sets.append(unique_labels.reshape(-1, 1))
            cursor += count

        return batch_label_sets

    @torch.no_grad()
    def predict_concepts_and_labels(
        self, dl, use_hard_logic=False, concept_refinement=True
    ):
        """
        Predicts both concept sets and label sets in a single pass.

        Returns:
            (all_concept_sets, all_label_sets)

            all_concept_sets: List of length N_samples. Each item is a list of arrays (one per concept).
            all_label_sets: List of length N_samples. Each item is an array of valid labels.
        """
        if self.per_concept_thresholds is None:
            raise ValueError("Run calibrate_per_concept first.")
        if not use_hard_logic and self.label_threshold is None:
            raise ValueError("Run calibrate_labels first (unless using hard logic).")

        self.model.eval()
        all_concept_tuples = []
        all_label_sets = []

        for data, _, _ in dl:
            data = data.to(self.device)

            # Forward Pass
            label_pred, conc_pred = self.model(data)

            # Build the initial "Conformal Tuples" (Cartesian Product)
            batch_marginal = self._build_concept_sets_for_batch(conc_pred)
            batch_tuples = []
            for sample_m in batch_marginal:
                # Handle empty sets by treating them as full range (wildcards)
                # TODO: for now it works in MNIST, what for BOIA and others?
                processed = [s if s.size > 0 else np.arange(self.concept_dim) for s in sample_m]
                sample_tuples = np.array(list(itertools.product(*processed)))
                batch_tuples.append(sample_tuples)
            
            all_concept_tuples.extend(batch_tuples)

            # Build Label Sets
            if use_hard_logic:
                batch_label_sets = self._compute_derived_labels_from_tuples(batch_tuples)
            else:
                # Standard Conformal Label prediction (1 - prob <= threshold)
                batch_label_sets = []
                for i in range(label_pred.size(0)):
                    if self.multilabel:
                        included = [
                            torch.where(1 - label_pred[i, j, :] <= self.label_threshold[j])[0].cpu().numpy()
                            for j in range(label_pred.size(1))
                        ]
                    else:
                        included = torch.where(1 - label_pred[i, :] <= self.label_threshold)[0].cpu().numpy()
                    batch_label_sets.append(included.reshape(-1, 1))
            all_label_sets.extend(batch_label_sets)

        # 3. Concept Refinement: Filter tuples based on predicted labels
        if not use_hard_logic and concept_refinement:
            all_concept_tuples = self._refine_concept_prediction_set(
                all_label_sets, all_concept_tuples
            )

        return all_concept_tuples, all_label_sets