import torch
import numpy as np
import itertools


class ConformalPredictor:
    def __init__(self, model, device, logic, concept_dim=10, n_concepts=2, multiconcepts=False, multilabel=False, bonferroni=False):
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
                        1 - conc_pred[:, i, j, :][range(concepts.size(0)), concepts[:, j].long()]
                        for i in range(self.n_concepts) for j in range(concepts.size(1))
                    ],
                    dim=1,
                )
            else:
                batch_scores = torch.stack(
                    [
                        1 - conc_pred[:, i, :][range(concepts.size(0)), concepts[:, i].long()]
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
                        1 - label_pred[:, i, :][range(labels.size(0)), labels[:, i].long()]
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
        k = scores.shape[1] # Number of concepts
        eff_alpha = alpha / k if self.bonferroni else alpha
        
        if self.bonferroni:
            print(f"[Conformal] Applying Bonferroni: Joint alpha {alpha} -> Per-concept alpha {eff_alpha:.4f}")

        # Compute the (1 - eff_alpha) quantile for each concept column
        self.per_concept_thresholds = np.quantile(scores, 1 - eff_alpha, axis=0)
        print(f"[Conformal] Per-concept thresholds: {self.per_concept_thresholds}")


    def calibrate_labels(self, dl, alpha=0.1):
        """
        Calibrate threshold for the final label set.
        """
        scores = self.compute_label_scores(dl)

        # Determine the effective alpha per label
        k = scores.shape[1] # Number of labels
        eff_alpha = alpha / k if self.bonferroni else alpha
        
        if self.bonferroni:
            print(f"[Conformal] Applying Bonferroni (Multilabel): Joint alpha {alpha} -> Per-label alpha {eff_alpha:.4f}")
        
        self.label_threshold = np.quantile(scores, 1 - eff_alpha, axis=0)
        print(f"[Conformal] Label threshold: {self.label_threshold}")

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
                                    1 - conc_pred[i, j, k, :][ :] <= self.per_concept_thresholds[k]
                                )[0]
                                .cpu()
                                .numpy()
                        )
                        included_list.append(included)
                    sample_set.append(included_list)
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
        prediction_sets = []

        for data, _, _ in dl:
            data = data.to(self.device)
            _, conc_pred = self.model(data)

            batch_sets = self._build_concept_sets_for_batch(conc_pred)
            prediction_sets.extend(batch_sets)

        return prediction_sets


    def _compute_derived_labels_vectorized(self, batch_concept_sets):
        """
        Private helper: Vectorized Hard Logic.
        1. Fills empty sets with range(10).
        2. Creates Cartesian products (meshgrid).
        3. Calls self.logic on the massive batch of combinations.
        """
        flat_concept_columns = []
        sample_counts = []

        for sample_sets in batch_concept_sets:
            # Fill empty sets with range
            cleaned_sets = [
                s if len(s) > 0 else np.arange(self.concept_dim) for s in sample_sets
            ]

            # Initialize storage on the first iteration
            if not flat_concept_columns:
                flat_concept_columns = [[] for _ in range(len(cleaned_sets))]

            # Create Cartesian Product efficiently using meshgrid
            # *cleaned_sets unpacks the list so meshgrid handles N concepts
            grids = np.meshgrid(*cleaned_sets, indexing="ij")
            for idx, g in enumerate(grids):
                flat_concept_columns[idx].append(g.ravel())

            sample_counts.append(len(grids[0].ravel()))

        if flat_concept_columns and len(flat_concept_columns[0]) > 0:
            big_cols = [np.concatenate(col_list) for col_list in flat_concept_columns]

            big_matrix = np.stack(big_cols, axis=1)

            # Call the logic ones
            raw_labels = self.logic.forward(big_matrix)

            if hasattr(raw_labels, "cpu"):
                raw_labels = raw_labels.cpu().numpy()

            raw_labels = np.array(raw_labels).ravel()
        else:
            raw_labels = np.array([])

        batch_label_sets = []
        cursor = 0

        for count in sample_counts:
            # Slice the logic outputs belonging to this sample
            sample_preds = raw_labels[cursor : cursor + count]

            unique_labels = np.unique(sample_preds)
            batch_label_sets.append(unique_labels)

            cursor += count

        return batch_label_sets

    def _refine_concept_prediction_set(
        self, label_prediction_set, concept_prediction_set
    ):
        """
        Refines concept prediction sets based on label predictions.

        Logic:
        1. Identify all concept tuples allowed by the predicted labels.
        2. Filter these tuples: keep only those compatible with the conformal concept sets
           (treating empty concept sets as wildcards/unknowns).
        3. Project the surviving tuples back to marginal concept sets.
        """
        refined_batch_concepts = []

        for labels, current_concept_sets in zip(
            label_prediction_set, concept_prediction_set
        ):

            if len(labels) == 0:
                refined_batch_concepts.append(current_concept_sets)
                continue

            # Get all the valid concepts for the labels
            allowed_tuples = set()
            for label in labels:
                implied = self.logic.get_concepts_for_label(label.item())
                allowed_tuples.update(implied)

            # Generate conformal tuples, where there are missing values
            processed_args = []
            for arr in current_concept_sets:
                if arr.size == 0:
                    processed_args.append(range(self.concept_dim))
                else:
                    processed_args.append(arr.tolist())

            # Get the conformal tuples
            conformal_tuples = list(itertools.product(*processed_args))
            # print(f"Conformal tuples: {conformal_tuples}")

            # filter the tuples
            valid_tuples = [t for t in conformal_tuples if t in allowed_tuples]
            n_concepts = len(current_concept_sets)

            if not valid_tuples:
                # return the empty set
                refined_sample = [np.array([]) for _ in range(n_concepts)]
            else:
                valid_matrix = np.array(valid_tuples)
                refined_sample = []
                for i in range(n_concepts):
                    unique_vals = np.unique(valid_matrix[:, i])
                    refined_sample.append(unique_vals)

            refined_batch_concepts.append(refined_sample)

        return refined_batch_concepts

    def _refine_label_prediction_set(
        self, concept_prediction_set, label_prediction_set
    ):
        """
        Refines label prediction sets based on concept predictions.
        """

        refined_batch_labels = []

        for labels, current_concept_sets in zip(
            label_prediction_set, concept_prediction_set
        ):

            # For the case of no labels only, get the most probable ones
            if len(labels) > 0:
                refined_batch_labels.append(labels)
                continue

            # Clean the concept values for no predicted ones
            cleaned_sets = [
                s if s.size > 0 else np.arange(self.concept_dim)
                for s in current_concept_sets
            ]

            # Generate Cartesian product
            grids = np.meshgrid(*cleaned_sets, indexing="ij")
            # Get the concept combinations in that case
            concept_combinations = np.stack([g.ravel() for g in grids], axis=1)

            # Inference via logic
            inferred_labels = self.logic.forward(concept_combinations)

            if hasattr(inferred_labels, "cpu"):
                inferred_labels = inferred_labels.cpu().numpy()

            # Get the labels back
            unique_inferred = np.unique(inferred_labels)
            refined_batch_labels.append(unique_inferred)

        return refined_batch_labels

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
        # Checks
        if self.per_concept_thresholds is None:
            raise ValueError("Run calibrate_per_concept first.")
        if not use_hard_logic and self.label_threshold is None:
            raise ValueError("Run calibrate_labels first (unless using hard logic).")

        self.model.eval()
        all_concept_sets = []
        all_label_sets = []

        for data, _, _ in dl:
            data = data.to(self.device)

            # 1. Forward Pass
            label_pred, conc_pred = self.model(data)

            # 2. Build Concept Sets
            batch_concept_sets = self._build_concept_sets_for_batch(conc_pred)
            all_concept_sets.extend(batch_concept_sets)

            # 3. Build Label Sets
            batch_label_sets = []
            if use_hard_logic:
                batch_label_sets = self._compute_derived_labels_vectorized(
                    batch_concept_sets
                )
            else:
                # Standard Conformal if not hard logic
                for i in range(label_pred.size(0)):
                    if self.multilabel:
                        included = []
                        for j in range(label_pred.size(1)):
                            incl = (
                                torch.where(
                                    1 - label_pred[i, j, :] <= self.label_threshold[j]
                                )[0]
                                .cpu()
                                .numpy()
                            )
                            included.append(incl)
                    else:
                        included = (
                            torch.where(1 - label_pred[i, :] <= self.label_threshold)[0]
                            .cpu()
                            .numpy()
                        )
                    
                    batch_label_sets.append(included)

            all_label_sets.extend(batch_label_sets)

        # 4. Optionally refine concept sets based on label sets
        if not use_hard_logic and concept_refinement:
            # Remove those concepts that are impossible for the labels
            all_concept_sets = self._refine_concept_prediction_set(
                all_label_sets, all_concept_sets
            )

            # 5. Optionally refine label sets based on concept sets
            if label_refinement:
                # Remove those concepts that are impossible for the labels
                all_label_sets = self._refine_label_prediction_set(
                    all_concept_sets, all_label_sets
                )

        return all_concept_sets, all_label_sets
