import torch
import numpy as np
from conformal.models.conformal import ConformalPredictor
from conformal.general_utils import log
from conformal.utils.alignment import (
    align_concepts,
    align_knowledge_input,
)
from functools import reduce


class ConformalEPredictor(ConformalPredictor):
    """
    Conformal predictor using e-values (soft-rank e-variables)
    instead of p-values / quantile thresholds.
    """

    def __init__(self, *args, alpha=0.1, beta=0.1, max_size_concepts=None, max_size_labels=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Chosen alpha and beta parameters for e-value computation
        self.alpha = alpha
        self.beta = beta

        # For storing sums of conformity scores for concepts and labels during calibration
        self.concept_score_sums = None
        self.label_score_sums = None

        # Maximum size of concept and label sets to consider for prediction (for computational efficiency)
        self.max_size_concepts = max_size_concepts
        self.max_size_labels = max_size_labels

        # Number of elements use to calibrate e-values (denominator of the average)
        self.n_concept_cal = None
        self.n_label_cal = None

    def compute_soft_rank_evalue(self, score, score_sum, n):
        """Compute the soft-rank e-value for a given conformity score."""
        return (n + 1) * score / (score_sum + score)
    

    @torch.no_grad()
    def compute_conformity_scores(self, dl):
        """
        Compute per-concept conformity scores on a calibration dataset.
        Conformity score = -log(P(c|x))

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

            # For each concept, compute - log(probability of true label)
            if self.multiconcepts:
                batch_scores = torch.stack(
                    [
                        -torch.log(
                            conc_pred[:, i, j, :][
                                range(concepts.size(0)), concepts[:, j].long()
                            ] + 1e-12
                        )
                        for i in range(self.n_concepts)
                        for j in range(concepts.size(1))
                    ],
                    dim=1,
                )
            else:
                batch_scores = torch.stack(
                    [
                        -torch.log(
                            conc_pred[:, i, :][
                                range(concepts.size(0)), concepts[:, i].long()
                            ] + 1e-12
                        )
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
        Conformity score = -log(P(y|x))
        """
        self.model.eval()
        all_scores = []

        for data, _, labels in dl:
            data, labels = data.to(self.device), labels.to(self.device)
            label_pred, _, _ = self.model(data, eval=True)

            if self.multilabel:
                batch_scores = torch.stack(
                    [
                        - torch.log(
                            label_pred[:, i, :][
                                range(labels.size(0)), labels[:, i].long()
                            ] + 1e-12
                        )
                        for i in range(labels.size(1))
                    ],
                    dim=1,
                )
            else:
                batch_scores = - torch.log(label_pred[range(len(labels)), labels.long()] + 1e-12)
            all_scores.append(batch_scores.cpu().numpy())

        return np.concatenate(all_scores)


    @torch.no_grad()
    def calibrate_per_concept(self, dl):
        """
        Calibrate concept e-values using conformity scores.

        Stores:
            self.concept_score_sums
            self.n_concept_cal
        """

        log("Computing conformity scores for concepts...", "INFO")

        scores = self.compute_conformity_scores(dl)

        self.concept_score_sums = scores.sum(axis=0)
        self.n_concept_cal = scores.shape[0]

        log(f"[E-CP] Concept calibration samples: {self.n_concept_cal}", "INFO")
        log(f"[E-CP] Concept score sums: {self.concept_score_sums}", "INFO")


    def compute_evalues(self, scores):
        """
        scores: shape (B, n_values, n_classes)
        """
        per_concept_e = []
        joint_evalues = []

        for i in range(scores.shape[1]):
            e_list = []
            for j in range(scores.shape[2]):
                e = self.compute_soft_rank_evalue(
                    scores[:, i, j],
                    self.concept_score_sums[i],
                    self.n_concept_cal,
                )
                e_list.append(e)
            per_concept_e.append(np.stack(e_list, axis=1)) # shape (B, n_classes)
        joint_evalues = np.stack(per_concept_e, axis=1)  # shape (B, n_values, n_classes)
        joint_evalues = np.array([
            reduce(np.kron, joint_evalues[b])
            for b in range(joint_evalues.shape[0])
        ])

        return joint_evalues


    def compute_label_evalues(self, scores):
        """scores: shape (B,) for single-label, (B, n_labels) for multi-label"""
        return self.compute_soft_rank_evalue(
            scores,
            self.label_score_sum,
            self.n_label_cal,
        )


    @torch.no_grad()
    def calibrate_labels(self, dl):
        """
        Calibrate label e-values using conformity scores.

        Stores:
            self.label_score_sum
            self.n_label_cal
        """

        log("Computing conformity scores for labels...", "INFO")

        scores = self.compute_label_scores(dl)

        if scores.ndim == 2:  # multilabel case
            self.label_score_sum = scores.sum(axis=0)
            self.n_label_cal = scores.shape[0]
        else:  # single label
            self.label_score_sum = scores.sum()
            self.n_label_cal = scores.shape[0]

        log(f"[E-CP] Label calibration samples: {self.n_label_cal}", "INFO")
        log(f"[E-CP] Label score sum: {self.label_score_sum}", "INFO")

    def _decode_indices(self, indices, n_concepts, n_classes):
        decoded = []
        for idx in indices:
            combo = []
            for _ in range(n_concepts):
                combo.append(int(idx % n_classes))
                idx //= n_classes
            decoded.append(np.array(combo[::-1]))  # reverse to get correct order
        return decoded

    def _build_sets_for_batch(self, conc_pred, beta, pad=False):
        scores = -torch.log(conc_pred.cpu() + 1e-12)  # shape (B, n_concepts, n_classes)
        evalues = self.compute_evalues(scores) # shape (B, n_classes)
        mask = evalues <= 1 / beta  # shape (B, n_classes)

        batch_sets = []
        n_concepts = conc_pred.shape[1]
        n_classes = conc_pred.shape[2]

        for b in range(mask.shape[0]):
            active = np.where(mask[b])[0]

            if len(active) == 0:
                batch_sets.append(np.expand_dims(np.array([self.EMPTY_TOKEN for _ in range(n_concepts)]), axis=0))
            else:
                decoded = self._decode_indices(active, n_concepts, n_classes)
                batch_sets.append(np.array(decoded))

        return batch_sets


    @torch.no_grad()
    def predict_concepts_and_labels(
        self,
        dl,
        alpha_labels,
        beta_concepts,
    ):
        """
        Predicts both concept sets and label sets in a single pass.

        Returns:
            (all_concept_sets, all_label_sets)

            all_concept_sets: List of length N_samples. Each item is a list of arrays (one per concept).
            all_label_sets: List of length N_samples. Each item is an array of valid labels.
        """

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
            batch_tuples = self._build_sets_for_batch(conc_pred, beta_concepts, pad=True)
            all_concept_tuples.extend(batch_tuples)

            if label_pred.ndim == 2:  # multi-label case
                label_pred = label_pred.unsqueeze(1)  # shape (B, 1, n_labels)

            # Build Label Sets
            batch_label_sets = self._build_sets_for_batch(
                label_pred, alpha_labels
            )
            all_label_sets.extend(batch_label_sets)

        all_concept_tuples = self._refine_concept_prediction_set(
            all_label_sets, all_concept_tuples
        )

        all_label_sets = self._refine_label_prediction_set(
            all_label_sets, all_concept_tuples
        )

        return all_concept_tuples, all_label_sets