import abc
import torch


class BaseLogic(abc.ABC):
    def __init__(self, n_concepts, concept_dim, is_too_big=False):
        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.is_too_big = is_too_big

    @abc.abstractmethod
    def forward(self, x):
        """Apply logic to concepts"""
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
    def __init__(self, logic_lambda, n_concepts, concept_dim, is_too_big=False):
        super().__init__(n_concepts, concept_dim, is_too_big)
        self.logic = logic_lambda

    def forward(self, x):
        return self.logic(x)
