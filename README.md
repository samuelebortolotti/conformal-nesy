# Conformal Prediction for Neuro-Symbolic Concept Bottleneck Models

This is the official codebase for the paper:

> **Concise and Logically Consistent Conformal Sets for Neuro-Symbolic Concept-Based Models**
> Recently accepted at **NeurIPS 2025** · [arXiv:2605.18202](https://arxiv.org/abs/2605.18202)

## Abstract

Neuro-Symbolic Concept-based Models (NeSy-CBMs) are a family of architectures that integrate neural networks with symbolic reasoning for enhanced reliability in high-stakes applications. They work by first extracting high-level concepts from the input and then inferring a task label from these compatibly with given logical constraints. Yet, their label and concept predictions can be overconfident, making it difficult for stakeholders to gauge when the model's decisions can be trusted. We address this issue by integrating ideas from Conformal Prediction (CP), a framework providing rigorous, distribution-free coverage guarantees. We formalize three desiderata — consistency, coverage, and conciseness — that any conformal method for NeSy-CBMs should satisfy, and show that existing approaches fall short of at least one. We then introduce COCOCO, a post-hoc framework that conformalizes concepts and labels jointly and reconciles them via a single deduction-abduction revision step. COCOCO satisfies all three desiderata, retains distribution-free coverage, is robust to imperfect knowledge and supports user-specified size budgets. Our experiments on 8 data sets highlight how COCOCO compares favorably against competitors and natural baselines in terms of performance and set size.

---

## Method

A NeSy-CBM has three components:

1. A **backbone encoder** (LeNet, ResNet-18, BERT, etc.) that maps raw inputs to concept probability distributions.
2. A **symbolic reasoning layer** that maps concept assignments to task labels via a fixed logic circuit — for example, `label = digit₁ + digit₂` for MNIST-Addition.
3. A **conformal wrapper** that uses a held-out calibration split to compute nonconformity thresholds and produce prediction sets.

Two NeSy backends are supported:

- **[DeepProbLog (DPL)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html)**.
- **[LogicTensorNetworks (LTN)](https://www.sciencedirect.com/science/article/pii/S0004370221002009)**.

Two conformal strategies are implemented:

- **P-value CP**.
- **E-value CP**.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Datasets

| ID | Task | Concepts | Notes |
|---|---|---|---|
| `mnistadd` | Sum of 2 MNIST digits (0–18) | 2 digit identities | Standard NeSy benchmark |
| `mnistaddn` | Sum of N MNIST digits | N digit identities | Scalability; set `--n-digits N` |
| `mnistsump` | Parity of the sum | 2 digit identities | |
| `mnisthalf` | Biased digit sum (0–8) | 2 digit identities | |
| `mnistevenodd` | Biased digit sum (0–18) | 2 digit identities | |
| `cifar` | CIFAR-10 object class | 7 binary visual attributes | Concept→class via logic |
| `rival` | RIVAL-10 species | 7 binary visual attributes | Concept→class via logic |
| `chx` | Chest X-ray pathology | 4 radiological findings | Medical imaging; requires NIH access |
| `cebab` | Restaurant review sentiment | 4 aspect scores | NLP; 5-class output |
| `boia` | Autonomous driving action | 21 road-scene concepts | 4 driving actions; see below |

### Downloading datasets

```bash
make download_mnist    # MNIST (via torchvision)
make download_cifar    # CIFAR-10 (via torchvision)
make download_cebab    # CeBaB (via Hugging Face Hub)
make download_rival    # RIVAL-10 (instructions printed)
make download_chx      # ChestX-ray NIH (instructions printed)
make download_all      # all auto-downloadable datasets
```

**RIVAL-10** must be downloaded from Kaggle (`rival10` dataset) and placed at `data/RIVAL10/`.

**ChestX-ray (NIH)** requires CSV annotation files from the [NIH CXR dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC). Place the following in `data/`:
- `four_findings_expert_labels_individual_readers.csv`
- `four_findings_expert_labels_test_labels.csv`
- `four_findings_expert_labels_validation_labels.csv`

Images are downloaded automatically by the loader on first use.

**BOIA (BDD-OIA)** uses the BDD-OIA dataset preprocessed by [rsbench](https://unitn-sml.github.io/rsbench/). Follow the rsbench data preparation instructions and place the resulting pickle files in `data/bdd2048/`.

---

## Running experiments

```
python -m conformal [--seed SEED] [--device {cuda,cpu}] OUTPUT_DIR COMMAND [OPTIONS] DATASET NETWORK NESY_MODEL
```

**Commands:** `train`, `test`, `etest`, `deltas`, `analyze`, `optuna`

**Datasets:** `mnistadd`, `mnistaddn`, `mnistsump`, `mnisthalf`, `mnistevenodd`, `cifar`, `rival`, `chx`, `cebab`, `boia`

**Networks:** `lenet`, `resnet18`, `linear`, `bert`, `mpnet`, `clip`

**NeSy models:** `dpl`, `ltn`, `dsl`, `linear_predictor`

### MNIST-Addition with DPL

```bash
python -m conformal --seed 1011 CONF_MNISTADD train \
    --learning-rate 0.1 --momentum 0.1 --batch-size 32 --opt sgd \
    --concept-sup 0.0 --epochs 200 \
    mnistadd lenet dpl

python -m conformal --seed 1011 CONF_MNISTADD test \
    --learning-rate 0.1 --momentum 0.1 --batch-size 32 --opt sgd \
    --concept-sup 0.0 --epochs 200 \
    mnistadd lenet dpl
```

### LTN operator configuration

```bash
python -m conformal --seed 1011 CONF_MNISTADD train \
    --learning-rate 0.0001 --momentum 0.9 --batch-size 256 --opt adam \
    --concept-sup 0.0 --epochs 200 \
    mnistadd lenet ltn \
    --and_op prod --or_op prod --imp_op goguen --p 4
```

Operator choices for `--and_op`, `--or_op`, `--imp_op`: `godel`, `prod`, `lukasiewicz`, `goguen`, `kleene`.

### Concept supervision

`--concept-sup` (float in [0, 1]) controls the fraction of training samples with ground-truth concept labels. `0.0` trains on task labels only; `1.0` gives full concept supervision.

### E-value conformal testing

```bash
python -m conformal --seed 1011 CONF_MNISTADD etest \
    --learning-rate 0.1 --batch-size 32 --opt sgd --concept-sup 0.0 --epochs 200 \
    mnistadd lenet dpl
```

### Hyperparameter search

```bash
python -m conformal --seed 42 CONF_OPT optuna mnistadd lenet ltn
```

---

## Output structure

```
OUTPUT_DIR/
├── <experiment_name>.best_model.pth       # Model checkpoint
├── <experiment_name>.train_results.csv    # Per-epoch training stats
└── <experiment_name>.test_results.csv     # Coverage / set-size / consistency
```

The experiment name encodes seed, dataset, network, NeSy model, and concept supervision level so runs with different configurations never collide.

---

## Citation

```bibtex
@misc{bortolotti2026concise,
    title={Concise and Logically Consistent Conformal Sets for Neuro-Symbolic Concept-Based Models}, 
    author={Samuele Bortolotti and Emanuele Marconato and Andrea Pugnana and Andrea Passerini and Stefano Teso},
    year={2026},
    eprint={2605.18202},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2605.18202}, 
}
```
