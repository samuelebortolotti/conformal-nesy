class Logic:
    def __init__(self, logic_lambda, n_concepts, concept_dim, is_too_big=False):
        self.logic = logic_lambda
        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.is_too_big = is_too_big

    def forward(self, x):
        return self.logic(x)
