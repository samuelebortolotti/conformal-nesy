# Conformal NeSy

This repository contains the code for our experiments on combining **conformal prediction** with **neuro-symbolic (NeSy) AI** methods. The idea is simple: instead of returning a single predicted label, we return a *prediction set* that is guaranteed to contain the true label with at least probability 1-α — and we do this on top of NeSy models (DeepProbLog and Logic Tensor Networks) that already give us **interpretable** and **consistent**, concept-level reasoning.

The experiments cover a range of tasks: digit arithmetic on `MNIST`, object recognition on `CIFAR-10` and `RIVAL-10`, `chest X-ray` pathology detection, skin lesion classification and sentiment analysis on restaurant reviews (`CEBaB`).

---

## How it works

The overall pipeline is:

1. A **backbone encoder** (LeNet, ResNet-18, BERT, etc.) maps raw inputs to concept-level predictions -- for example, on MNIST-Addition the encoder predicts which digit each image shows.
2. A **NeSy wrapper** uses those concept predictions to reason about the task label through a symbolic logic circuit — the label is determined by a fixed rule (e.g. `label = digit_1 + digit_2`).
3. A **conformal wrapper** is applied at test time: using a held-out calibration set, we compute a nonconformity threshold and use it to produce prediction sets with provable marginal coverage guarantees.

Two NeSy backends are supported:

- **DeepProbLog (DPL)**: Implements probabilistic logic via a probabilistic circuit. Concept predictions are independent Bernoulli/categorical variables; label probabilities are computed via weighted-model counting over all consistent "worlds". Inference is exact and deterministic.
- **Logic Tensor Networks (LTN)**: Implements the logic in fuzzy semantics using neural operators (Gödel, product, Łukasiewicz, etc.). The logic satisfaction is part of the training loss, so the model learns to satisfy the rules softly.

For conformal prediction we implement two approaches:

- **Standard threshold-based conformal**: calibrate a quantile threshold on nonconformity scores; include all labels whose score is below that threshold.
- **E-value based conformal**: an alternative using soft-rank e-variables, which can behave better on small calibration sets.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# optional dev tools 
pip install -r requirements.dev.txt
```

The main dependencies are PyTorch, TorchVision, LTNtorch, Optuna, scikit-learn, seaborn, and medmnist. See [requirements.txt](requirements.txt) for the pinned versions.

---

## Running experiments

The CLI is the `conformal` module:

```
python -m conformal [--seed SEED] [--device {cuda,cpu}] OUTPUT_DIR COMMAND [OPTIONS] DATASET NETWORK NESY_MODEL
```

**Commands:** `train`, `test`, `etest`, `deltas`, `analyze`, `optuna`, `joint_failure`

**Datasets:** `mnistadd`, `mnistsump`, `mnisthalf`, `mnistevenodd`, `mnistaddn`, `cifar`, `rival`, `cebab`, `chx`

**Networks:** `lenet`, `resnet18`, `linear`, `bert`, `lama`

**NeSy models:** `dpl`, `ltn`

### Quick start — MNIST-Addition with DPL

```bash
# Train
python -m conformal --seed 1011 CONF_MNIST train \
    --epochs 200 \
    --learning-rate 0.1 \
    --momentum 0.1 \
    --batch-size 32 \
    --opt sgd \
    --concept-sup 0.0 \
    mnistadd lenet dpl

# Evaluate with conformal prediction
python -m conformal --seed 1011 CONF_MNIST test \
    --epochs 200 \
    --learning-rate 0.1 \
    --momentum 0.1 \
    --batch-size 32 \
    --opt sgd \
    --concept-sup 0.0 \
    mnistadd lenet dpl
```

The test command wraps the trained model in a conformal predictor, calibrates on the validation split, and reports coverage, set size, and consistency on the test set.

### LTN — configuring the fuzzy operators

LTN has several operator choices you can tune:

```bash
python -m conformal --seed 1011 CONF_MNIST train \
    --epochs 200 \
    --learning-rate 0.1 \
    --momentum 1e-5 \
    --batch-size 64 \
    --opt sgd \
    --concept-sup 0.0 \
    mnistadd lenet ltn \
    --and_op prod \
    --or_op prod \
    --imp_op prod \
    --p 8
```

Operator choices for `--and_op`, `--or_op`, `--imp_op`: `godel`, `prod`, `lukasiewicz`, `goguen`, `kleene`. `--p` controls the quantifier aggregation exponent.

### MNIST with N digits

The `mnistaddn` dataset generalises MNIST-Addition to N digits:

```bash
python -m conformal --seed 1011 CONF_MNIST train \
    --concept-sup 0.0 --epochs 200 --learning-rate 0.1 \
    --batch-size 32 --opt sgd \
    mnistaddn --n-digits 3 \
    lenet dpl
```

### Concept supervision

The `--concept-sup` flag controls what fraction of training samples come with ground-truth concept labels (digit identities, attribute flags, etc.). At `0.0` the model is trained only on task labels; at `1.0` it gets full concept supervision. This lets you study the effect of weak/no concept labels on conformal efficiency.

### ChestX-ray

```bash
python -m conformal --seed 1011 CONF_CHX train \
    --epochs 50 --learning-rate 1e-4 --batch-size 32 \
    --concept-sup 1.0 \
    chx resnet18 dpl --chx-multi-class

python -m conformal --seed 1011 CONF_CHX test \
    --epochs 50 --learning-rate 1e-4 --batch-size 32 \
    --concept-sup 1.0 \
    chx resnet18 dpl --chx-multi-class
```

### CEBaB (text)

```bash
python -m conformal --seed 1011 CONF_CEBAB train \
    --epochs 20 --learning-rate 2e-5 --batch-size 16 \
    --concept-sup 1.0 \
    cebab bert dpl

python -m conformal --seed 1011 CONF_CEBAB test \
    --epochs 20 --learning-rate 2e-5 --batch-size 16 \
    --concept-sup 1.0 \
    cebab bert dpl
```

### E-value based conformal

```bash
python -m conformal --seed 1011 CONF_MNIST etest \
    --epochs 200 --learning-rate 0.1 --batch-size 32 --opt sgd --concept-sup 0.0 \
    mnistadd lenet dpl
```

### Hyperparameter optimization

```bash
python -m conformal --seed 1011 CONF_OPT optuna \
    mnistadd lenet ltn
```

Uses Optuna to tune LTN operator choices, learning rate, etc.

---

## Output structure

Each run saves its results under `OUTPUT_DIR/`:

```
OUTPUT_DIR/
├── <experiment_name>.best_model.pth       # Trained model checkpoint
├── <experiment_name>.train_results.csv    # Per-epoch training stats
└── <experiment_name>.test_results.csv     # Conformal evaluation metrics
```

The experiment name is derived automatically from the command-line arguments (seed, dataset, network, NeSy model, concept supervision, etc.), so results from different configurations don't collide.

---

## Datasets

| Dataset | Task | Concepts | Notes |
|---|---|---|---|
| `mnistadd` | Sum of two MNIST digits (0–18) | Two digit identities | Classic NeSy benchmark |
| `mnistaddn` | Sum of N MNIST digits | N digit identities | Scalability test; set `--n-digits` |
| `mnistsump` | Parity of the sum | Two digits | JRSs benchmark |
| `mnisthalf` | Biased sum of two MNIST digits (0-8) | Two digits | RSs benchmark |
| `mnistevenodd` | Biased sum of two MNIST digits (0-18) | Two digits | RSs benchmark |
| `cifar` | CIFAR-10 object class | Binary visual attributes | Concept to class via logic |
| `rival` | RIVAL-10 species | Binary visual attributes | Concept to class via logic |
| `chx` | Chest X-ray pathology | 4 radiological findings | Medical; `--chx-multi-class` for 5-class |
| `cebab` | Restaurant review sentiment | 4 aspect scores | Text; 5-class sentiment |

---

## Misc options

- `--device cuda` / `--device cpu` — default is `cuda`
- `--seed INT` — sets random seed for full reproducibility (data splits, weight init, etc.)
- `--dry-run` — run the full pipeline without writing any output files (useful for debugging)
- `--output-compression {7z,gzip}` — compress output CSVs
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`
