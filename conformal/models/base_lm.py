import torch
import torch.nn as nn

from transformers import AutoModel
from conformal.models.mlp import MLP


class BaseLM(nn.Module):

    def __init__(
        self,
        model_name="bert-base-uncased",
        num_concept_dim=4,
        num_concepts=5,
        freeze_encoder=True,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        hidden_size = self.encoder.config.hidden_size
        self.num_concept_dim = num_concept_dim
        self.num_concepts = num_concepts

        self.classifier = MLP(
            input_dim=hidden_size,
            num_classes=self.num_concept_dim * self.num_concepts,
        )

    def encode(self, input_ids, attention):
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention,
                )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention,
            )

        # class token
        pooled = outputs.last_hidden_state[:, 0]

        return pooled

    def forward(self, x):
        input_ids, attention = x[:, :, 0], x[:, :, 1]
        embeddings = self.encode(input_ids, attention)
        logits = self.classifier(embeddings)

        logits = logits.view(x.shape[0], self.num_concepts, self.num_concept_dim)
        return logits

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        else:
            self.encoder.train(mode)
        self.classifier.train(mode)
        return self

    def eval(self):
        super().eval()
        self.encoder.eval()
        self.classifier.eval()
        return self
