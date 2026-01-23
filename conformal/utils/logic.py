import numpy as np
import itertools


class Logic:
    def __init__(self, logic_lambda, n_concepts, concept_dim, is_too_big=False):
        self.logic = logic_lambda
        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.is_too_big = is_too_big
        if self.is_too_big:
            self.label_concept_map = None
        else:
            self.label_concept_map = self._initialize_label_concept_map(
                n_concepts, concept_dim
            )

    def _initialize_label_concept_map(self, n_concepts, concept_dim):
        combinations = np.array(
            list(itertools.product(range(concept_dim), repeat=n_concepts))
        )
        predictions = self.forward(combinations)
        predictions = predictions.reshape(-1)
        label_concept_map = {}
        for comb, pred in zip(combinations, predictions):
            if pred not in label_concept_map:
                label_concept_map[pred] = set()
            label_concept_map[pred].add(tuple(comb.tolist()))
        return label_concept_map

    def forward(self, x):
        return self.logic(x)

    def get_concepts_for_label(self, target_label):
        return self.label_concept_map.get(target_label, set())
