import torch
import numpy as np
import itertools
from conformal.general_utils import log
from conformal.utils.alignment import (
    align_concepts,
    align_knowledge_input,
)


class ConformalPredictor:
    def __init__(
        self,
        model,
        device,
        logic,
        dataset,
        experiment_name,
        concept_dim=10,
        n_concepts=2,
        multiconcepts=False,
        multilabel=False,
        bonferroni=False,
        kernelize=False,
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
        self.dataset = dataset
        self.experiment_name = experiment_name

        self.per_concept_thresholds = None
        self.label_threshold = None
        self.bonferroni = bonferroni
        self.kernelize = kernelize
        self.permutation = None
        self.EMPTY_TOKEN = -1

    @torch.no_grad()
    def compute_permutation(self, dl):
        """Compute concept permutation"""
        self.model.eval()

        all_probs = []
        all_labels = []

        for data, concepts, _ in dl:
            data = data.to(self.device)
            concepts = concepts.cpu().numpy()

            _, conc_pred, _ = self.model(data, eval=True)
            conc_pred = conc_pred.cpu().numpy()

            all_probs.append(conc_pred)
            all_labels.append(concepts)

        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        _, permutation = align_concepts(
            probs if self.dataset.startswith("mnist") else probs.squeeze(1),
            labels,
            multiclass=self.multiconcepts,
        )

        log(f"[Conformal] Learned concept permutation:\n{permutation}", "INFO")
        return permutation

    def set_permutation(self, permutation):
        """Setting permutation"""
        self.permutation = permutation

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
            _, conc_pred, _ = self.model(data, eval=True)

            if self.permutation is not None:
                log(
                    "Applying permutation to concept predictions for conformity score computation...",
                    "INFO",
                )
                conc_pred = align_knowledge_input(
                    (
                        conc_pred.detach().cpu().numpy()
                        if self.dataset.startswith("mnist")
                        else conc_pred.squeeze(1).detach().cpu().numpy()
                    ),
                    self.permutation,
                )
                conc_pred = torch.tensor(conc_pred, device=self.device)
                conc_pred = (
                    conc_pred
                    if self.dataset.startswith("mnist")
                    else conc_pred.unsqueeze(1)
                )

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
            label_pred, _, _ = self.model(data, eval=True)

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
                "INFO",
            )
        else:
            log(
                f"[Conformal] Calibrating per-concept thresholds with alpha {alpha} (no Bonferroni adjustment)",
                "INFO",
            )

        # Compute the (1 - eff_alpha) quantile for each concept column
        self.per_concept_thresholds = np.quantile(scores, 1 - eff_alpha, axis=0)
        log(
            f"[Conformal] Per-concept thresholds: {self.per_concept_thresholds}", "INFO"
        )

        # SANITY CHECK
        import matplotlib.pyplot as plt

        for i in range(k):
            concept_scores = scores[:, i]
            q = self.per_concept_thresholds[i]

            plt.figure()

            # Histogram of scores
            plt.hist(concept_scores, bins=50)

            # Quantile line
            plt.axvline(q)

            plt.title(f"Concept {i} score distribution")
            plt.xlabel("Nonconformity score")
            plt.ylabel("Frequency")
            plt.savefig(f"{self.experiment_name}_concept_{i}_scores.pdf")
            plt.close()

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
                "INFO",
            )
        else:
            log(
                f"[Conformal] Calibrating label threshold with alpha {alpha} (no Bonferroni adjustment)",
                "INFO",
            )

        self.label_threshold = np.quantile(scores, 1 - eff_alpha, axis=0)
        log(f"[Conformal] Label threshold: {self.label_threshold}", "INFO")

        # SANITY CHECK

        import matplotlib.pyplot as plt

        for i in range(k):
            label_scores = scores if len(scores.shape) == 1 else scores[:, i]
            q = self.label_threshold

            plt.figure()

            # Histogram of scores
            plt.hist(label_scores, bins=50)

            # Quantile line
            plt.axvline(q)

            plt.title(f"Label {i} score distribution")
            plt.xlabel("Nonconformity score")
            plt.ylabel("Frequency")
            plt.savefig(f"{self.experiment_name}_label_{i}_scores.pdf")
            plt.close()

    def _build_concept_sets_for_batch(self, conc_pred):

        # aggregate the thresholds
        thresholds = torch.tensor(self.per_concept_thresholds, device=conc_pred.device)

        # build the scores
        scores = 1 - conc_pred

        if self.multiconcepts:

            thresholds = thresholds.view(1, 1, -1, 1)
            mask = scores <= thresholds

            batch_sets = []

            # get the batch
            for sample_mask in mask:
                included_list = [
                    torch.where(sample_mask[0, k])[0].cpu().numpy()
                    for k in range(sample_mask.shape[1])
                ]
                batch_sets.append(included_list)

            return batch_sets

        else:
            thresholds = thresholds.view(1, -1, 1)
            mask = scores <= thresholds

            # get the batch
            batch_sets = [
                [
                    torch.where(sample_mask[j])[0].cpu().numpy()
                    for j in range(sample_mask.shape[0])
                ]
                for sample_mask in mask
            ]

            return batch_sets

    def _generate_combinations(self, marginal_list):
        """Generate Cartesian product of concept sets, handling empty sets with a placeholder token."""
        processed = []
        for s in marginal_list:
            if len(s) == 0:
                processed.append(np.array([self.EMPTY_TOKEN], dtype=int))
            else:
                processed.append(s)
        return np.array(list(itertools.product(*processed)), dtype=int)

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
            _, conc_pred, _ = self.model(data, eval=True)

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
            if tuples.size == 0:
                # Keep tuples as is when empty
                refined_batch_tuples.append(tuples)
                continue

            # empty tuples mask
            empty_mask = (tuples == self.EMPTY_TOKEN).any(axis=1)

            if len(labels) == 0:
                # keep only those containing the EMPTY_TOKEN
                filtered_tuples = tuples[empty_mask]
                refined_batch_tuples.append(filtered_tuples)
                continue

            if self.permutation is not None:
                tuples = (
                    torch.nn.functional.one_hot(
                        torch.tensor(tuples),
                        self.concept_dim if self.dataset.startswith("mnist") else 2,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                tuples = align_knowledge_input(tuples, self.permutation)
                tuples = np.argmax(tuples, axis=-1)

            # Filter them
            filtered_tuples = tuples[~empty_mask]
            derived_labels = self.logic.forward(filtered_tuples)

            # Keep tuples that result in an allowed label
            valid_tuples = filtered_tuples[np.isin(derived_labels, labels)]
            refined_batch_tuples.append(valid_tuples)

        return refined_batch_tuples

    def _refine_label_prediction_set(
        self, label_prediction_set, concept_prediction_tuples
    ):
        """
        Refines label tuples based on concept predictions.
        Returns only the filtered tuples.
        """
        refined_label_sets = []

        for labels, tuples in zip(label_prediction_set, concept_prediction_tuples):
            if len(tuples) == 0 or len(labels) == 0:
                refined_label_sets.append(np.array([], dtype=int))
                continue

            # empty tuples mask
            empty_mask = np.any(tuples == self.EMPTY_TOKEN, axis=1)

            if self.permutation is not None:
                tuples = (
                    torch.nn.functional.one_hot(
                        torch.tensor(tuples),
                        self.concept_dim if self.dataset.startswith("mnist") else 2,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                tuples = align_knowledge_input(tuples, self.permutation)
                tuples = np.argmax(tuples, axis=-1)

            filtered_tuple = tuples[~empty_mask]
            derived_labels = self.logic.forward(filtered_tuple)

            # Keep labels that are produced by a tuple
            valid_mask = np.isin(labels, derived_labels)
            valid_labels = np.expand_dims(labels[valid_mask], axis=1)
            refined_label_sets.append(valid_labels)

        return refined_label_sets

    def _apply_logic(self, concept_matrix):
        if concept_matrix.size == 0:
            return np.array([], dtype=int)

        # EMPTY rows containing EMPTY_TOKEN in any concept column are invalid
        invalid_mask = (concept_matrix == self.EMPTY_TOKEN).any(axis=1)
        valid_mask = ~invalid_mask

        raw_labels = np.empty(concept_matrix.shape[0], dtype=object)

        # Process valid rows
        if np.any(valid_mask):
            valid_matrix = concept_matrix[valid_mask]

            if self.permutation is not None:
                valid_matrix = (
                    torch.nn.functional.one_hot(
                        torch.tensor(valid_matrix),
                        self.concept_dim if self.dataset.startswith("mnist") else 2,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                valid_matrix = align_knowledge_input(valid_matrix, self.permutation)
                valid_matrix = np.argmax(valid_matrix, axis=-1)

            labels_valid = self.logic.forward(valid_matrix)
            if hasattr(labels_valid, "cpu"):
                labels_valid = labels_valid.cpu().numpy()

            labels_valid = np.array(labels_valid).ravel()
            raw_labels[valid_mask] = labels_valid

        # Assign empty arrays for invalid tuples
        raw_labels[invalid_mask] = None
        return raw_labels

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

        # Track indices to split the batch later
        sample_counts = [t.shape[0] for t in batch_concept_tuples]

        # Flatten all tuples into one massive matrix for a single logic pass
        big_matrix = np.concatenate(batch_concept_tuples, axis=0)

        if self.permutation is not None:
            big_matrix = (
                torch.nn.functional.one_hot(
                    torch.tensor(big_matrix),
                    self.concept_dim if self.dataset.startswith("mnist") else 2,
                )
                .detach()
                .cpu()
                .numpy()
            )
            big_matrix = align_knowledge_input(
                big_matrix,
                self.permutation,
            )
            big_matrix = np.argmax(big_matrix, axis=-1)

        # Apply logic to the big matrix
        raw_labels = self._apply_logic(big_matrix)

        # Ensure labels are flat (for single-label classification tasks)
        raw_labels = np.array(raw_labels).ravel()

        # Reconstruct per-sample unique label sets
        batch_label_sets = []
        cursor = 0
        for count in sample_counts:
            # Slice labels belonging to this specific sample
            sample_preds = raw_labels[cursor : cursor + count]
            # remove the None
            clean_preds = np.array([x for x in sample_preds if x is not None])
            unique_labels = np.unique(clean_preds)
            if unique_labels.size == 1 and unique_labels[0] is None:
                unique_labels = np.array([])
            batch_label_sets.append(unique_labels.reshape(-1, 1))
            cursor += count

        return batch_label_sets

    @torch.no_grad()
    def predict_labels(self, label_pred, batch_tuples=None, use_hard_logic=False):
        """
        Builds conformal label prediction sets.
        """

        if use_hard_logic:
            if batch_tuples is None:
                raise ValueError("batch_tuples required when use_hard_logic=True")

            batch_label_sets = self._compute_derived_labels_from_tuples(batch_tuples)
            return batch_label_sets

        # Standard conformal label prediction
        if self.label_threshold is None:
            raise ValueError("Run calibrate_labels first.")

        scores = 1 - label_pred

        threshold = torch.tensor(self.label_threshold, device=label_pred.device)

        if self.multilabel:
            # label_pred shape: (B, N_labels, N_classes)
            thresholds = threshold.view(1, -1, 1)

            mask = scores <= thresholds

            batch_label_sets = [
                [
                    torch.where(mask[b, j])[0].unsqueeze(1).cpu().numpy()
                    for j in range(mask.shape[1])
                ]
                for b in range(mask.shape[0])
            ]

        else:
            # label_pred shape: (B, N_classes)
            mask = scores <= threshold

            batch_label_sets = [
                torch.where(mask[b])[0].unsqueeze(1).cpu().numpy()
                for b in range(mask.shape[0])
            ]

        return batch_label_sets

    @torch.no_grad()
    def predict_concepts_and_labels(
        self, dl, use_hard_logic=False, concept_refinement=True, label_refinement=False
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
            label_pred, conc_pred, _ = self.model(data, eval=True)

            if self.permutation is not None:
                conc_pred = align_knowledge_input(
                    (
                        conc_pred.detach().cpu().numpy()
                        if self.dataset.startswith("mnist")
                        else conc_pred.squeeze(1).detach().cpu().numpy()
                    ),
                    self.permutation,
                )
                conc_pred = torch.tensor(conc_pred, device=self.device)
                conc_pred = (
                    conc_pred
                    if self.dataset.startswith("mnist")
                    else conc_pred.unsqueeze(1)
                )

            # Build the initial "Conformal Tuples" (Cartesian Product)
            batch_marginal = self._build_concept_sets_for_batch(conc_pred)

            batch_tuples = []
            for sample_m in batch_marginal:
                sample_tuples = self._generate_combinations(sample_m)
                batch_tuples.append(sample_tuples)

            all_concept_tuples.extend(batch_tuples)

            # Build Label Sets
            batch_label_sets = self.predict_labels(
                label_pred, batch_tuples=batch_tuples, use_hard_logic=use_hard_logic
            )
            all_label_sets.extend(batch_label_sets)

        # 3. Concept Refinement: Filter tuples based on predicted labels
        if not use_hard_logic and concept_refinement:
            all_concept_tuples = self._refine_concept_prediction_set(
                all_label_sets, all_concept_tuples
            )

        if not use_hard_logic and label_refinement:
            # Refine label sets based on concept tuples (not implemented here)
            all_label_sets = self._refine_label_prediction_set(
                all_label_sets, all_concept_tuples
            )

        return all_concept_tuples, all_label_sets
