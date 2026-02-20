import os
import pandas as pd
import numpy as np
import torch
import json

from transformers import AutoTokenizer
from torch.utils.data import Dataset

from conformal.utils.logic import HardLogic
from conformal.models import bert, lama, mpnet


def configure_global_arguments(parser):
    pass


class CeBaBDataset(Dataset):
    """Wrap CeBaB text + concepts + labels."""

    def __init__(
        self,
        texts,
        concepts,
        labels,
        tokenizer_name="bert-base-uncased",
        max_length=128,
    ):
        self.texts = texts
        self.concepts = torch.tensor(concepts, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

        self.tokenized_samples = []
        for input_ids, attention_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            merged_input = torch.stack([input_ids, attention_mask], axis=1)
            self.tokenized_samples.append(merged_input)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            self.tokenized_samples[idx],
            self.concepts[idx],
            self.labels[idx],
        )


class CeBaBLoader:
    def __init__(self, tokenizer_name="bert-base-uncased", data_dir="data/cebab", device="cuda"):
        self.data_dir = data_dir
        self.device = device
        self.tokenizer_name=tokenizer_name

        # Aspects and states
        self.aspects = ["food", "service", "noise", "ambiance"]
        self.states = ["Positive", "Negative", "Neutral", "unknown"]
        self.concepts = ["food", "service", "noise", "ambiance", "rating"]
        self.labels = ["positive", "negative", "neutral", "unknown", "conflict"]
        self.n_concepts = len(self.concepts)
        self.n_labels = len(self.labels)
        self.concept_dim = len(self.aspects)
        self.label_weights = []
        self.concepts_weights = []

    def build_concepts(self, entries):
        """Convert JSON entries to one-hot concept matrix"""
        concepts_list = []
        for aspect in self.aspects:
            vals = []
            for e in entries:
                v = e.get(f"{aspect}_aspect_majority", "unknown")
                if not v.strip() or v == "no majority":
                    v = "unknown"
                vals.append(v)

            current_aspect_list = []
            for v in vals:
                current_aspect_list.append(self.states.index(v))
            concepts_list.append(current_aspect_list)
        # append rating, last concept
        concepts_list.append(self.build_star_concept_labels(entries))
        return np.stack(concepts_list, axis=1)

    def build_star_concept_labels(self, entries):
        """Convert stars entries to concept"""
        concept = np.zeros(len(entries))
        for i, e in enumerate(entries):
            r = e.get("review_majority", "unknown")
            if r == "no majority" or r == "unknown":
                concept[i] = 1  # unknown
            elif r in ["1", "2"]:
                concept[i] = 1  # Negative
            elif r in ["3"]:
                concept[i] = 1  # Neutral
            elif r in ["4", "5"]:
                concept[i] = 1  # Positive
            else:
                raise ValueError(f"Unexpected review_majority value: {r}")
        return concept

    def _return_cebab_logic(self):
        def build_labels(concepts):
            labels = []

            for sample_states in concepts:
                n_pos = np.sum(sample_states == 0)
                n_neg = np.sum(sample_states == 1)
                n_neu = np.sum(sample_states == 2)
                n_unk = np.sum(sample_states == 3)

                # majority is both positive and negative
                if n_pos >= max(n_neu, n_unk) and n_pos == n_neg:
                    labels.append(4)
                # majority positive or (positive-unk, positive-neut)
                elif n_pos >= max(n_neg, n_neu, n_unk):
                    labels.append(0)
                # majority negative or (negative-unk, negative-neut)
                elif n_neg >= max(n_pos, n_neu, n_unk):
                    labels.append(1)
                # majority neutral
                elif n_neu >= max(n_pos, n_neg, n_unk):
                    labels.append(2)
                # majority unknown
                else:
                    labels.append(3)

            return np.array(labels, dtype=np.int64)

        return build_labels

    def load(self):
        train_path = os.path.join(self.data_dir, "train_observational.json")
        dev_path = os.path.join(self.data_dir, "dev.json")
        test_path = os.path.join(self.data_dir, "test.json")

        with open(train_path, "r", encoding="utf-8") as f:
            train_entries = json.load(f)
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_entries = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_entries = json.load(f)

        # Texts
        X_train = [e["description"] for e in train_entries]
        X_dev = [e["description"] for e in dev_entries]
        X_test = [e["description"] for e in test_entries]

        # Concepts
        C_train = self.build_concepts(train_entries)
        C_dev = self.build_concepts(dev_entries)
        C_test = self.build_concepts(test_entries)

        build_labels = self._return_cebab_logic()

        # Labels
        y_train = build_labels(C_train)
        y_dev = build_labels(C_dev)
        y_test = build_labels(C_test)

        y_counts = np.bincount(y_train, minlength=self.n_labels)

        self.label_weights = torch.tensor(
            [
                len(y_train) / (self.n_labels * count) if count > 0 else 1.0
                for count in y_counts
            ], 
            dtype=torch.float
        ).to(self.device)

        for i in range(self.n_concepts):
            concept_counts = np.bincount(C_train[:, i].astype(int), minlength=self.concept_dim)
            self.concepts_weights.append(torch.tensor(
                [
                    len(C_train) / (self.concept_dim * count) if count > 0 else 1.0
                    for count in concept_counts
                ],
                dtype=torch.float32,
            ))
        self.concepts_weights = torch.stack(self.concepts_weights, dim=0).to(self.device)

        # logic
        logic = HardLogic(
            self._return_cebab_logic(),
            n_concepts=self.n_concepts,
            concept_dim=self.concept_dim,
        )

        # Wrap in Dataset
        train_ds = CeBaBDataset(X_train, C_train, y_train, tokenizer_name=self.tokenizer_name)
        dev_ds = CeBaBDataset(X_dev, C_dev, y_dev, tokenizer_name=self.tokenizer_name)
        test_ds = CeBaBDataset(X_test, C_test, y_test, tokenizer_name=self.tokenizer_name)

        return (
            train_ds,
            dev_ds,
            test_ds,
            None,
            self.concept_dim,
            self.n_labels,
            self.n_concepts,
            self.labels,
            self.states,
            logic,
            torch.nn.CrossEntropyLoss(weight=self.label_weights),
            self.concepts_weights,
            self.label_weights,
        )


def configure_subparsers(subparsers):
    cebab_parser = subparsers.add_parser(
        "cebab",
        help="Use CEBAB dataset",
    )

    configure_global_arguments(cebab_parser)

    sub = cebab_parser.add_subparsers(dest="model")
    bert.configure_subparsers(sub)
    lama.configure_subparsers(sub)
    mpnet.configure_subparsers(sub)
