import abc
import itertools
import torch
import numpy as np
from conformal.general_utils import log


class BaseLogic(abc.ABC):
    def __init__(self, n_concepts, concept_dim, is_too_big=False):
        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.is_too_big = is_too_big
        self.label_to_world_dict = {}
        self.EMPTY_TOKEN = -1

    @abc.abstractmethod
    def forward(self, x):
        """Apply logic to concepts"""
        raise NotImplementedError

    @abc.abstractmethod
    def abductive_concept_sets(self, y):
        """Get concept sets that would lead to label y"""
        raise NotImplementedError


def _random_argmax(out):
    """Return argmax with uniform random tie-breaking among tied maximizers."""
    out_np = out.detach().cpu().numpy()
    max_vals = out_np.max(axis=-1, keepdims=True)
    tied = out_np == max_vals
    return np.array([np.random.choice(np.where(row)[0]) for row in tied])


class LinearLayerLogic(BaseLogic):

    def __init__(self, model, n_concepts, concept_dim, is_too_big=False):
        super().__init__(n_concepts, concept_dim, is_too_big)
        self.model = model
        self.multi_set_logic = None

    def forward(self, x):
        concepts = self.model._from_predictions_to_probabilities(x)
        concepts = self.model._normalize(concepts)
        out, _ = self.model.inference(concepts)
        return _random_argmax(out)

    def _build_label_to_world_dict(self):
        if self.is_too_big:
            log("World space too large to enumerate.", "INFO")
            return self.label_to_world_dict

        self.label_to_world_dict = {}
        values = list(range(self.concept_dim))

        if getattr(self.model, "dataset", "") != "boia":
            # Invert the linear layer directly: for each concept world compute the
            # feature index matching outer_product's layout (f = k1*M^(N-1)+…+kN)
            # and read argmax(W[:, f] + b). This avoids log(~0) numerical issues
            # that occur when running the full forward pass on one-hot inputs.
            linear_layer = self.model.linear[0]
            W = linear_layer.weight.detach().cpu().numpy()  # [output_dim, feature_dim]
            b = linear_layer.bias.detach().cpu().numpy()    # [output_dim]
            for world in itertools.product(values, repeat=self.n_concepts):
                f = 0
                for c in world:
                    f = f * self.concept_dim + c
                label = int(np.argmax(W[:, f] + b))
                if label not in self.label_to_world_dict:
                    self.label_to_world_dict[label] = []
                self.label_to_world_dict[label].append(np.array(world))

            # Fallback: for labels not reachable by the column-argmax (e.g. a CBM
            # trained on soft probabilities may never assign those labels to any
            # discrete one-hot feature), use the feature column that most strongly
            # activates label y and decode it to a concept world.
            M = self.concept_dim
            for y in range(W.shape[0]):
                if y not in self.label_to_world_dict:
                    best_f = int(np.argmax(W[y, :] + b[y]))
                    remaining = best_f
                    world_list = []
                    for _ in range(self.n_concepts):
                        world_list.append(remaining % M)
                        remaining //= M
                    world_arr = np.array(world_list[::-1])
                    self.label_to_world_dict[y] = [world_arr]

            return self.label_to_world_dict

        # BOIA uses additive/concatenated features rather than outer product;
        # fall back to forward-pass enumeration.
        for world in itertools.product(values, repeat=self.n_concepts):
            world_array = np.expand_dims(np.array(world), axis=0)
            label = self.forward(world_array)

            if label.shape[0] > 1:
                for el in label:
                    if el.item() not in self.label_to_world_dict:
                        self.label_to_world_dict[el.item()] = []
                    self.label_to_world_dict[el.item()].append(np.array(world))
            else:
                if label.item() not in self.label_to_world_dict:
                    self.label_to_world_dict[label.item()] = []
                self.label_to_world_dict[label.item()].append(np.array(world))

        return self.label_to_world_dict

    def abductive_concept_sets(self, y):
        concept_sets = []

        if self.is_too_big:
            log(
                "World space too large to enumerate. Abductive concept sets not available.",
                "WARNING",
            )
            return concept_sets

        if not self.label_to_world_dict:
            log("Label to world dictionary not built yet. Building now...", "INFO")
            self.label_to_world_dict = self._build_label_to_world_dict()

        for pred_y in y:
            if pred_y.size == 0:
                concept_sets.append(
                    np.array([[self.EMPTY_TOKEN for _ in range(self.n_concepts)]])
                )
            else:
                vals = []
                for el_y in pred_y:
                    key = el_y.item()
                    if key not in self.label_to_world_dict:
                        log(
                            f"Label {key} not reachable by any concept world; skipping for abduction.",
                            "WARNING",
                        )
                        continue
                    vals.append(self.label_to_world_dict[key])
                if vals:
                    concept_sets.append(np.vstack(vals))
                else:
                    concept_sets.append(
                        np.array([[self.EMPTY_TOKEN for _ in range(self.n_concepts)]])
                    )
        return concept_sets


class DSLLogic(BaseLogic):
    def __init__(
        self,
        model,
        n_concepts,
        concept_dim,
        is_too_big=False,
    ):
        super().__init__(n_concepts, concept_dim, is_too_big)
        self.model = model
        self.multi_set_logic = None

    def forward(self, x):
        concepts = self.model._from_predictions_to_probabilities(x)
        concepts = self.model._normalize(concepts)
        out, _ = self.model.inference(concepts, eval=True)
        return _random_argmax(out)

    def _build_label_to_world_dict(self):
        if self.is_too_big:
            log("World space too large to enumerate.", "INFO")
            return self.label_to_world_dict

        values = list(range(self.concept_dim))
        worlds = itertools.product(values, repeat=self.n_concepts)

        self.label_to_world_dict = {}

        for world in worlds:
            world_array = np.expand_dims(np.array(world), axis=0)
            label = self.forward(world_array)

            if label.shape[0] > 1:
                for el in label:
                    if el.item() not in self.label_to_world_dict:
                        self.label_to_world_dict[el.item()] = []
                    self.label_to_world_dict[el.item()].append(np.array(world))
            else:
                if label.item() not in self.label_to_world_dict:
                    self.label_to_world_dict[label.item()] = []
                self.label_to_world_dict[label.item()].append(np.array(world))

        return self.label_to_world_dict

    def abductive_concept_sets(self, y):
        concept_sets = []

        if self.is_too_big:
            log(
                "World space too large to enumerate. Abductive concept sets not available.",
                "WARNING",
            )
            return concept_sets

        if not self.label_to_world_dict:
            log("Label to world dictionary not built yet. Building now...", "INFO")
            self.label_to_world_dict = self._build_label_to_world_dict()

        for pred_y in y:
            if pred_y.size == 0:
                concept_sets.append(
                    np.array([[self.EMPTY_TOKEN for _ in range(self.n_concepts)]])
                )
            else:
                vals = []
                for el_y in pred_y:
                    key = el_y.item()
                    if key not in self.label_to_world_dict:
                        log(
                            f"Label {key} not reachable by any concept world; skipping for abduction.",
                            "WARNING",
                        )
                        continue
                    vals.append(self.label_to_world_dict[key])
                if vals:
                    concept_sets.append(np.vstack(vals))
                else:
                    concept_sets.append(
                        np.array([[self.EMPTY_TOKEN for _ in range(self.n_concepts)]])
                    )
        return concept_sets


class HardLogic(BaseLogic):
    def __init__(
        self,
        logic_lambda,
        n_concepts,
        concept_dim,
        is_too_big=False,
        multi_set_logic=None,
    ):
        super().__init__(n_concepts, concept_dim, is_too_big)
        self.logic = logic_lambda
        self.multi_set_logic = multi_set_logic
        self.n_labels = None  # set by subclass if multilabel

    def forward(self, x):
        return self.logic(x)

    def forward_multi_set(self, x):
        if self.multi_set_logic is not None:
            return self.multi_set_logic(x)
        else:
            return self.logic(x)

    def _build_label_to_world_dict(self):
        if self.is_too_big:
            log("World space too large to enumerate.", "INFO")
            return self.label_to_world_dict

        values = list(range(self.concept_dim))
        worlds = itertools.product(values, repeat=self.n_concepts)

        self.label_to_world_dict = {}

        for world in worlds:
            world_array = np.expand_dims(np.array(world), axis=0)
            # wrong logic case
            if self.multi_set_logic is not None:
                derived = self.multi_set_logic(world_array)
                label = np.unique(np.concatenate(derived))
            else:
                label = self.logic(world_array)

            if label.shape[0] > 1:
                for el in label:
                    if el.item() not in self.label_to_world_dict:
                        self.label_to_world_dict[el.item()] = []
                    self.label_to_world_dict[el.item()].append(np.array(world))
            else:
                if label.item() not in self.label_to_world_dict:
                    self.label_to_world_dict[label.item()] = []
                self.label_to_world_dict[label.item()].append(np.array(world))

        return self.label_to_world_dict

    def abductive_concept_sets(self, y):
        concept_sets = []

        if self.is_too_big:
            log(
                "World space too large to enumerate. Abductive concept sets not available.",
                "WARNING",
            )
            return concept_sets

        if not self.label_to_world_dict:
            log("Label to world dictionary not built yet. Building now...", "INFO")
            self.label_to_world_dict = self._build_label_to_world_dict()

        for pred_y in y:
            if pred_y.size == 0:
                concept_sets.append(
                    np.array([[self.EMPTY_TOKEN for _ in range(self.n_concepts)]])
                )
            else:
                vals = []
                for el_y in pred_y:
                    key = el_y.item()
                    if key not in self.label_to_world_dict:
                        log(
                            f"Label {key} not reachable by any concept world; skipping for abduction.",
                            "WARNING",
                        )
                        continue
                    vals.append(self.label_to_world_dict[key])
                if vals:
                    concept_sets.append(np.vstack(vals))
                else:
                    concept_sets.append(
                        np.array([[self.EMPTY_TOKEN for _ in range(self.n_concepts)]])
                    )
        return concept_sets


class FactorizedBoiaLogic(HardLogic):
    """
    Factorized world enumeration for BOIA (21 concepts, 3 independent groups).

    STOP and FORWARD depend only on FS concepts (indices 0-8, 2^9 = 512 subworlds).
    LEFT  depends only on L concepts  (indices 9,10,11,18,19,20, 2^6 = 64 subworlds).
    RIGHT depends only on R concepts  (indices 12-17, 2^6 = 64 subworlds).

    Total worlds to evaluate: 640 instead of 2^21 ≈ 2M.
    """

    FS_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    L_IDX  = [9, 10, 11, 18, 19, 20]
    R_IDX  = [12, 13, 14, 15, 16, 17]
    MAX_ABDUCTION_WORLDS = 2048
    N_LABELS = 4

    def __init__(self, logic_lambda, n_concepts=1, concept_dim=21, multi_set_logic=None):
        super().__init__(
            logic_lambda,
            n_concepts=n_concepts,
            concept_dim=concept_dim,
            is_too_big=False,
            multi_set_logic=multi_set_logic,
        )
        self.n_labels = self.N_LABELS
        self._fs_cache = None  # (stop, forward) -> list of 9-dim binary arrays
        self._l_cache  = None  # left            -> list of 6-dim binary arrays
        self._r_cache  = None  # right           -> list of 6-dim binary arrays

    def forward(self, x):
        """Return binary (N, 4) labels, binarizing the soft BOIA logic output."""
        probs = self.logic(x)
        return (np.array(probs) > 0).astype(int)

    def _ensure_cache(self):
        if self._fs_cache is not None:
            return

        self._fs_cache = {}
        for bits in itertools.product(range(2), repeat=len(self.FS_IDX)):
            vec = np.zeros(21, dtype=int)
            for k, gi in enumerate(self.FS_IDX):
                vec[gi] = bits[k]
            labels = self.logic(vec.reshape(1, -1))
            stop    = int(labels[0, 0] > 0)
            forward = int(labels[0, 1] > 0)
            key = (stop, forward)
            self._fs_cache.setdefault(key, []).append(np.array(bits, dtype=int))

        self._l_cache = {}
        for bits in itertools.product(range(2), repeat=len(self.L_IDX)):
            vec = np.zeros(21, dtype=int)
            for k, gi in enumerate(self.L_IDX):
                vec[gi] = bits[k]
            labels = self.logic(vec.reshape(1, -1))
            left = int(labels[0, 2] > 0)
            self._l_cache.setdefault(left, []).append(np.array(bits, dtype=int))

        self._r_cache = {}
        for bits in itertools.product(range(2), repeat=len(self.R_IDX)):
            vec = np.zeros(21, dtype=int)
            for k, gi in enumerate(self.R_IDX):
                vec[gi] = bits[k]
            labels = self.logic(vec.reshape(1, -1))
            right = int(labels[0, 3] > 0)
            self._r_cache.setdefault(right, []).append(np.array(bits, dtype=int))

    def _build_label_to_world_dict(self):
        # Not used for multilabel BOIA; abductive_concept_sets overrides directly.
        return {}

    def _assemble_vec(self, fs_bits, l_bits, r_bits):
        vec = np.zeros(21, dtype=int)
        for k, gi in enumerate(self.FS_IDX):
            vec[gi] = fs_bits[k]
        for k, gi in enumerate(self.L_IDX):
            vec[gi] = l_bits[k]
        for k, gi in enumerate(self.R_IDX):
            vec[gi] = r_bits[k]
        return vec

    def abductive_concept_sets(self, y):
        """
        Return per-sample concept sets consistent with predicted label sets.

        y: list of (K, 4) ndarrays -- multilabel label sets (each row is a 4-bit vector)
        Returns: list of 3-tuples (fs_worlds, l_worlds, r_worlds) where each element is a
                 2D array of shape (n_worlds, group_size) containing group sub-vectors.
                 This matches the format returned by _generate_combinations_boia so that
                 boia_conformal_metrics and _prediction_consistency_boia_groups apply.
        """
        self._ensure_cache()
        concept_sets = []

        empty_fs = np.zeros((0, len(self.FS_IDX)), dtype=int)
        empty_l  = np.zeros((0, len(self.L_IDX)),  dtype=int)
        empty_r  = np.zeros((0, len(self.R_IDX)),  dtype=int)

        for pred_y in y:
            pred_y = np.asarray(pred_y)
            if pred_y.size == 0:
                concept_sets.append((empty_fs, empty_l, empty_r))
                continue
            if pred_y.ndim == 1:
                pred_y = pred_y.reshape(1, -1)

            # Collect needed group behaviors from every label vector in the set
            fs_keys, l_keys, r_keys = set(), set(), set()
            for lv in pred_y:
                fs_keys.add((int(lv[0]), int(lv[1])))
                l_keys.add(int(lv[2]))
                r_keys.add(int(lv[3]))

            fs_worlds = [w for key in fs_keys for w in self._fs_cache.get(key, [])]
            l_worlds  = [w for key in l_keys  for w in self._l_cache.get(key,  [])]
            r_worlds  = [w for key in r_keys  for w in self._r_cache.get(key,  [])]

            if not fs_worlds or not l_worlds or not r_worlds:
                concept_sets.append((empty_fs, empty_l, empty_r))
                continue

            # Return per-group sub-vectors as 2D arrays (no Cartesian product needed)
            fs_arr = np.array(fs_worlds, dtype=int)  # (n_fs, 9)
            l_arr  = np.array(l_worlds,  dtype=int)  # (n_l, 6)
            r_arr  = np.array(r_worlds,  dtype=int)  # (n_r, 6)
            concept_sets.append((fs_arr, l_arr, r_arr))

        return concept_sets
