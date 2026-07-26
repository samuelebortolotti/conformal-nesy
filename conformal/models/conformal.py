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

        if self.dataset == "boia" and self.bonferroni:
            # BOIA: group-specific Bonferroni correction
            # FS group (indices 0-8): 9 concepts
            # L group (indices 9,10,11,18,19,20): 6 concepts
            # R group (indices 12-17): 6 concepts
            log(
                f"[Conformal] BOIA group-specific Bonferroni: alpha={alpha}",
                "INFO",
            )
            per_concept_alpha = np.zeros(k)
            for i in self.BOIA_FS_IDX:
                per_concept_alpha[i] = alpha / len(self.BOIA_FS_IDX)
            for i in self.BOIA_L_IDX:
                per_concept_alpha[i] = alpha / len(self.BOIA_L_IDX)
            for i in self.BOIA_R_IDX:
                per_concept_alpha[i] = alpha / len(self.BOIA_R_IDX)
            self.per_concept_thresholds = np.array([
                np.quantile(scores[:, i], 1 - per_concept_alpha[i])
                for i in range(k)
            ])
        else:
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
        # import matplotlib.pyplot as plt

        # for i in range(k):
        #     concept_scores = scores[:, i]
        #     q = self.per_concept_thresholds[i]

        #     plt.figure()

        #     # Histogram of scores
        #     plt.hist(concept_scores, bins=50)

        #     # Quantile line
        #     plt.axvline(q)

        #     plt.title(f"Concept {i} score distribution")
        #     plt.xlabel("Nonconformity score")
        #     plt.ylabel("Frequency")
        #     plt.savefig(f"{self.experiment_name}_concept_{i}_scores.pdf")
        #     plt.close()

    def calibrate_labels(self, dl, alpha=0.1):
        """
        Calibrate threshold for the final label set.

        For BOIA with Bonferroni: uses per-group correction so each independent
        group (FS k=2, L k=1, R k=1) achieves ≥(1-alpha) marginal coverage.
        """
        scores = self.compute_label_scores(dl)

        k = 1 if len(scores.shape) == 1 else scores.shape[1]

        if self.dataset == "boia" and self.bonferroni and k == 4:
            # Per-group Bonferroni: labels [0,1]=FS(k=2), [2]=L(k=1), [3]=R(k=1)
            eff_alphas = [alpha / 2, alpha / 2, alpha, alpha]
            self.label_threshold = np.array([
                np.quantile(scores[:, i], 1 - eff_alphas[i]) for i in range(4)
            ])
            log(
                f"[Conformal] BOIA per-group label Bonferroni: "
                f"FS eff_alpha={alpha/2:.4f}, L eff_alpha={alpha:.4f}, R eff_alpha={alpha:.4f}",
                "INFO",
            )
        else:
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

        # import matplotlib.pyplot as plt

        # for i in range(k):
        #     label_scores = scores if len(scores.shape) == 1 else scores[:, i]
        #     q = self.label_threshold

        #     plt.figure()

        #     # Histogram of scores
        #     plt.hist(label_scores, bins=50)

        #     # Quantile line
        #     plt.axvline(q)

        #     plt.title(f"Label {i} score distribution")
        #     plt.xlabel("Nonconformity score")
        #     plt.ylabel("Frequency")
        #     plt.savefig(f"{self.experiment_name}_label_{i}_scores.pdf")
        #     plt.close()

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

    MAX_CONCEPT_TUPLES = 8192

    # BOIA factorized group indices (must match FactorizedBoiaLogic)
    BOIA_FS_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    BOIA_L_IDX  = [9, 10, 11, 18, 19, 20]
    BOIA_R_IDX  = [12, 13, 14, 15, 16, 17]

    def _generate_combinations(self, marginal_list):
        """
        Generate concept worlds from per-concept marginal sets.

        For BOIA: enumerate each factorized group (FS/L/R) independently,
        producing at most 2^9 + 2^6 + 2^6 = 640 worlds instead of 2^21.
        Each world row has EMPTY_TOKEN for concepts outside its group.

        For other datasets: standard Cartesian product, capped at MAX_CONCEPT_TUPLES.
        """
        if self.dataset == "boia":
            return self._generate_combinations_boia(marginal_list)

        processed = []
        for s in marginal_list:
            if len(s) == 0:
                processed.append(np.array([self.EMPTY_TOKEN], dtype=int))
            else:
                processed.append(s)

        total = 1
        for s in processed:
            total *= len(s)

        if total <= self.MAX_CONCEPT_TUPLES:
            return np.array(list(itertools.product(*processed)), dtype=int)

        rng = np.random.default_rng()
        rows = np.stack(
            [rng.choice(s, size=self.MAX_CONCEPT_TUPLES, replace=True) for s in processed],
            axis=1,
        )
        return np.unique(rows, axis=0)

    def _generate_combinations_boia(self, marginal_list):
        """
        Factorized world enumeration for BOIA.

        Enumerates FS / L / R groups independently (max 512 + 64 + 64 worlds).
        Returns a 3-tuple (fs_worlds, l_worlds, r_worlds) where each element is a
        2D array of shape (n_worlds, group_size) containing the group sub-vectors.
        """
        group_arrays = []
        for group_idx in [self.BOIA_FS_IDX, self.BOIA_L_IDX, self.BOIA_R_IDX]:
            group_size = len(group_idx)
            group_marginals = [marginal_list[i] for i in group_idx]

            # If any concept's marginal is empty, that group has 0 worlds
            if any(len(s) == 0 for s in group_marginals):
                group_arrays.append(np.zeros((0, group_size), dtype=int))
                continue

            combos = list(itertools.product(*group_marginals))
            group_arrays.append(np.array(combos, dtype=int))

        return tuple(group_arrays)

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

            # Get marginals first
            batch_marginal_sets = self._build_concept_sets_for_batch(conc_pred)

            # Convert to tuples immediately if we can
            for sample_marginal in batch_marginal_sets:
                tuples = self._generate_combinations(sample_marginal)
                all_tuple_sets.append(tuples)

        return all_tuple_sets

    def _refine_boia_concept_groups(self, labels, group_tuple):
        """
        Filter BOIA per-group concept sub-worlds by label consistency.

        group_tuple is (fs_worlds, l_worlds, r_worlds) where each is shape (n, group_size).
        For each group, keep only rows whose sub-vector produces a label output
        consistent with at least one label in the predicted set.
        """
        fs_worlds, l_worlds, r_worlds = group_tuple

        if len(labels) == 0:
            return group_tuple  # empty label set → keep all rows unchanged

        labels_arr = np.asarray(labels)
        if labels_arr.ndim == 1:
            labels_arr = labels_arr.reshape(1, -1)

        req_stop_fwd = set(map(tuple, labels_arr[:, :2].tolist()))
        req_left  = set(int(v) for v in labels_arr[:, 2])
        req_right = set(int(v) for v in labels_arr[:, 3])

        group_specs = [
            (self.BOIA_FS_IDX, fs_worlds, req_stop_fwd, lambda lv: (int(lv[1] > 0), int(lv[0] > 0))),
            (self.BOIA_L_IDX,  l_worlds,  req_left,     lambda lv: int(lv[2] > 0)),
            (self.BOIA_R_IDX,  r_worlds,  req_right,    lambda lv: int(lv[3] > 0)),
        ]

        result_groups = []
        for grp_idx, grp_worlds, req_vals, extract_fn in group_specs:
            group_size = len(grp_idx)
            if len(grp_worlds) == 0:
                result_groups.append(np.zeros((0, group_size), dtype=int))
                continue
            keep = []
            for row in grp_worlds:
                eval_vec = np.zeros(21, dtype=int)
                for local_i, gi in enumerate(grp_idx):
                    eval_vec[gi] = row[local_i]
                lv = np.asarray(self.logic.forward(eval_vec.reshape(1, -1)))
                lv = lv[0] if lv.ndim == 2 else lv.ravel()
                if extract_fn(lv) in req_vals:
                    keep.append(row)
            if keep:
                result_groups.append(np.array(keep, dtype=int))
            else:
                result_groups.append(np.zeros((0, group_size), dtype=int))

        return tuple(result_groups)

    def _refine_boia_label_groups(self, labels, group_tuple):
        """
        Filter BOIA label vectors to only those achievable by the factorized group sub-worlds.

        group_tuple is (fs_worlds, l_worlds, r_worlds) where each is shape (n, group_size).
        Computes achievable (stop,fwd)/left/right values per group and keeps label
        vectors where all bits are achievable independently.
        """
        fs_worlds, l_worlds, r_worlds = group_tuple
        n_labels = getattr(self.logic, "n_labels", 4)

        if len(labels) == 0:
            return np.zeros((0, n_labels), dtype=int)

        labels_arr = np.asarray(labels)
        if labels_arr.ndim == 1:
            labels_arr = labels_arr.reshape(1, -1)

        achievable_stop_fwd = set()
        achievable_left     = set()
        achievable_right    = set()

        group_specs = [
            (self.BOIA_FS_IDX, fs_worlds, achievable_stop_fwd, lambda lv: (int(lv[1] > 0), int(lv[0] > 0))),
            (self.BOIA_L_IDX,  l_worlds,  achievable_left,     lambda lv: int(lv[2] > 0)),
            (self.BOIA_R_IDX,  r_worlds,  achievable_right,    lambda lv: int(lv[3] > 0)),
        ]
        for grp_idx, grp_worlds, achievable_set, extract_fn in group_specs:
            for row in grp_worlds:
                eval_vec = np.zeros(21, dtype=int)
                for local_i, gi in enumerate(grp_idx):
                    eval_vec[gi] = row[local_i]
                lv = np.asarray(self.logic.forward(eval_vec.reshape(1, -1)))
                lv = lv[0] if lv.ndim == 2 else lv.ravel()
                achievable_set.add(extract_fn(lv))

        valid_mask = np.array([
            (int(lv[0]), int(lv[1])) in achievable_stop_fwd
            and int(lv[2]) in achievable_left
            and int(lv[3]) in achievable_right
            for lv in labels_arr
        ], dtype=bool)

        if not np.any(valid_mask):
            return np.zeros((0, n_labels), dtype=int)
        return labels_arr[valid_mask]

    def _refine_concept_prediction_set(
        self, label_prediction_set, concept_prediction_tuples
    ):
        """
        Refines concept tuples based on label predictions.
        Returns only the filtered tuples.
        """
        refined_batch_tuples = []

        for labels, tuples in zip(label_prediction_set, concept_prediction_tuples):
            # BOIA factorized path: group_tuple is a 3-tuple of sub-world arrays
            if isinstance(tuples, tuple):
                refined_batch_tuples.append(self._refine_boia_concept_groups(labels, tuples))
                continue

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
            if len(filtered_tuples) == 0:
                refined_batch_tuples.append(filtered_tuples)
                continue
            if self.logic.multi_set_logic is None:
                derived_labels = self.logic.forward(filtered_tuples)
                # Keep tuples that result in an allowed label
                if self.multilabel and np.asarray(derived_labels).ndim == 2 and np.asarray(labels).ndim == 2:
                    labels_arr = np.asarray(labels)
                    valid_rows = np.array([
                        np.any(np.all(dr == labels_arr, axis=1))
                        for dr in derived_labels
                    ], dtype=bool)
                    valid_tuples = filtered_tuples[valid_rows]
                else:
                    valid_tuples = filtered_tuples[np.isin(derived_labels, labels)]
                refined_batch_tuples.append(valid_tuples)
            else:
                derived_labels = self.logic.forward_multi_set(filtered_tuples)
                valid_mask = np.array(
                    [
                        len(np.intersect1d(row_labels, labels)) > 0
                        for row_labels in derived_labels
                    ],
                    dtype=bool,
                )
                valid_tuples = filtered_tuples[valid_mask]
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
            # BOIA factorized path: group_tuple is a 3-tuple of sub-world arrays
            if isinstance(tuples, tuple):
                refined_label_sets.append(self._refine_boia_label_groups(labels, tuples))
                continue

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
            if len(filtered_tuple) == 0:
                refined_label_sets.append(np.array([], dtype=int))
                continue
            if self.logic.multi_set_logic is None:
                derived_labels = self.logic.forward(filtered_tuple)
            else:
                derived_labels_sets = self.logic.forward_multi_set(filtered_tuple)
                derived_labels = np.unique(np.concatenate(derived_labels_sets))

            # Keep labels that are produced by a tuple
            derived_labels_arr = np.asarray(derived_labels)
            labels_arr = np.asarray(labels)
            if self.multilabel and derived_labels_arr.ndim == 2 and labels_arr.ndim == 2:
                valid_mask = np.array([
                    np.any(np.all(lbl == derived_labels_arr, axis=1))
                    for lbl in labels_arr
                ], dtype=bool)
                valid_labels = labels_arr[valid_mask]  # (K', n_labels)
                refined_label_sets.append(valid_labels)
            else:
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

            labels_valid = np.array(labels_valid)
            if self.multilabel and labels_valid.ndim == 2:
                # multilabel: store each row as an object so raw_labels stays (N,)
                for j, idx in enumerate(np.where(valid_mask)[0]):
                    raw_labels[idx] = labels_valid[j]
            else:
                raw_labels[valid_mask] = labels_valid.ravel()

        # Assign empty arrays for invalid tuples
        raw_labels[invalid_mask] = None
        return raw_labels

    def _compute_derived_labels_from_boia_group_tuples(self, batch_concept_tuples):
        """
        Derives labels from BOIA group-tuple concept sets using Hard Logic.

        Each element of batch_concept_tuples is a 3-tuple (fs_worlds, l_worlds, r_worlds).
        Returns list of (K, 4) unique label arrays per sample.
        """
        n_labels = getattr(self.logic, "n_labels", 4)
        batch_label_sets = []
        group_idx_list = [self.BOIA_FS_IDX, self.BOIA_L_IDX, self.BOIA_R_IDX]
        for group_tuple in batch_concept_tuples:
            # Collect achievable label bits from each group
            achievable_stop_fwd = set()
            achievable_left     = set()
            achievable_right    = set()
            achievable_sets = [achievable_stop_fwd, achievable_left, achievable_right]
            extract_fns = [
                lambda lv: (int(lv[0] > 0), int(lv[1] > 0)),
                lambda lv: int(lv[2] > 0),
                lambda lv: int(lv[3] > 0),
            ]

            for grp_idx, grp_worlds, ach_set, extract_fn in zip(
                group_idx_list, group_tuple, achievable_sets, extract_fns
            ):
                for row in grp_worlds:
                    eval_vec = np.zeros(21, dtype=int)
                    for local_i, gi in enumerate(grp_idx):
                        eval_vec[gi] = row[local_i]
                    lv = np.asarray(self.logic.forward(eval_vec.reshape(1, -1)))
                    lv = lv[0] if lv.ndim == 2 else lv.ravel()
                    ach_set.add(extract_fn(lv))

            # Build all achievable label combinations
            label_combos = []
            for sf in achievable_stop_fwd:
                for left in achievable_left:
                    for right in achievable_right:
                        label_combos.append([sf[0], sf[1], left, right])

            if label_combos:
                unique_labels = np.unique(np.array(label_combos, dtype=int), axis=0)
            else:
                unique_labels = np.zeros((0, n_labels), dtype=int)
            batch_label_sets.append(unique_labels)
        return batch_label_sets

    def _compute_derived_labels_from_tuples(self, batch_concept_tuples):
        """
        Derives labels from concept tuples using Hard Logic.

        Args:
            batch_concept_tuples: List of np.ndarrays, each of shape (N_combinations, N_concepts)
                                  OR list of 3-tuples for BOIA group representation.
        Returns:
            List of np.ndarrays containing unique predicted labels per sample.
        """
        if not batch_concept_tuples:
            return []

        # BOIA group-tuple path
        if isinstance(batch_concept_tuples[0], tuple):
            return self._compute_derived_labels_from_boia_group_tuples(batch_concept_tuples)

        # Track indices to split the batch later
        sample_counts = [t.shape[0] for t in batch_concept_tuples]

        # Flatten all tuples into one massive matrix for a single logic pass
        big_matrix = np.concatenate(batch_concept_tuples, axis=0)

        # Apply logic to the big matrix (_apply_logic handles EMPTY_TOKEN filtering
        # and the permutation/one-hot transform internally on valid rows only)
        raw_labels = self._apply_logic(big_matrix)

        # raw_labels is an object array (N_total,); for multilabel each entry is a (4,) array
        raw_labels = np.array(raw_labels)

        # Reconstruct per-sample unique label sets
        batch_label_sets = []
        cursor = 0
        for count in sample_counts:
            sample_preds = raw_labels[cursor : cursor + count]
            valid_preds = [x for x in sample_preds if x is not None]
            if self.multilabel:
                if valid_preds:
                    clean_preds = np.stack(valid_preds)  # (count, n_labels)
                    unique_labels = np.unique(clean_preds, axis=0)  # (K, n_labels)
                else:
                    n_labels = getattr(self.logic, "n_labels", 1)
                    unique_labels = np.zeros((0, n_labels), dtype=int)
                batch_label_sets.append(unique_labels)
            else:
                clean_preds = np.array(valid_preds)
                unique_labels = np.unique(clean_preds)
                if unique_labels.size == 1 and unique_labels[0] is None:
                    unique_labels = np.array([])
                batch_label_sets.append(unique_labels.reshape(-1, 1))
            cursor += count

        return batch_label_sets

    @torch.no_grad()
    def predict_labels(self, dl, batch_tuples=None, use_hard_logic=False):
        """
        Predict conformal label sets for all samples in a dataloader.

        Returns:
            all_label_sets[sample] = array/list of included labels
        """

        self.model.eval()
        all_label_sets = []
        tuple_idx = 0

        for data, _, _ in dl:
            data = data.to(self.device)

            # forward pass
            label_pred, _, _ = self.model(data, eval=True)

            if use_hard_logic:
                if batch_tuples is None:
                    raise ValueError("batch_tuples required when use_hard_logic=True")

                current_batch_tuples = batch_tuples[
                    tuple_idx : tuple_idx + data.size(0)
                ]
                batch_label_sets = self.predict_label_set(
                    label_pred=None,
                    batch_tuples=current_batch_tuples,
                    use_hard_logic=True,
                )
                tuple_idx += data.size(0)
            else:
                batch_label_sets = self.predict_label_set(label_pred)

            all_label_sets.extend(batch_label_sets)

        return all_label_sets

    @torch.no_grad()
    def predict_label_set(self, label_pred, batch_tuples=None, use_hard_logic=False):
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
            mask = scores <= thresholds  # (B, N_labels, N_classes)
            batch_label_sets = []
            for b in range(mask.shape[0]):
                per_label_classes = [
                    torch.where(mask[b, j])[0].cpu().numpy()
                    for j in range(mask.shape[1])
                ]
                prod = list(itertools.product(*per_label_classes))
                if prod:
                    combos = np.array(prod, dtype=int)  # (K, N_labels)
                else:
                    combos = np.zeros((0, mask.shape[1]), dtype=int)
                batch_label_sets.append(combos)

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
        self,
        dl,
        use_hard_logic=False,
        concept_refinement=True,
        label_refinement=False,
        abduction=False,
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

            # do the kernel later only when required
            for sample_m in batch_marginal:
                sample_tuples = self._generate_combinations(sample_m)
                batch_tuples.append(sample_tuples)
            all_concept_tuples.extend(batch_tuples)

            # Build Label Sets
            batch_label_sets = self.predict_label_set(
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

        if abduction:
            # For each sample, determine which concepts could produce the predicted label
            all_concept_tuples = self.logic.abductive_concept_sets(all_label_sets)

        return all_concept_tuples, all_label_sets
