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


class LinearLayerLogic(BaseLogic):

    def __init__(self, model, n_concepts, concept_dim, is_too_big=False):
        super().__init__(n_concepts, concept_dim, is_too_big)
        self.model = model

    def forward(self, x):
        concepts = self.model._from_predictions_to_probabilities(x)
        concepts = self.model._normalize(concepts)
        out, _ = self.model.inference(concepts)
        return torch.argmax(out, dim=-1).detach().cpu().numpy()


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

    def forward(self, x):
        concepts = self.model._from_predictions_to_probabilities(x)
        concepts = self.model._normalize(concepts)
        out, _ = self.model.inference(concepts, eval=True)
        return torch.argmax(out, dim=-1).detach().cpu().numpy()


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
                    assert (
                        key in self.label_to_world_dict
                    ), f"Missing key in label to concept dictionary for abduction: {key}"
                    vals.append(self.label_to_world_dict[key])
                concept_sets.append(np.vstack(vals))
        return concept_sets
