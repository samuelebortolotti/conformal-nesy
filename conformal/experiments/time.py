"""Measure per-example inference latency for each conformal method."""

import time as _time  # alias immediately — this file is named time.py

import csv
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from conformal.general_utils import get_basename, log
from conformal.utils.factories import (
    DatasetFactory,
    NetworkFactory,
    NeSyFactory,
    LogicFactory,
)
from conformal.datasets.loaders import create_dataloaders
from conformal.experiments.utils import load_model
from conformal.models.conformal import ConformalPredictor
from conformal.models.econformal import ConformalEPredictor
from conformal.datasets import (
    boia,
    chx,
    derma,
    mnistadd,
    mnisthalf,
    mnistsump,
    mnistaddn,
    mnistevenodd,
    cifar,
    rival,
    cebab,
)


def configure_global_arguments(parser):
    """Configure global arguments shared across models and datasets."""
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for training."
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Optimizer momentum."
    )
    parser.add_argument(
        "--concept-supervision",
        type=float,
        default=1.0,
        help="Concept supervision weight.",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="sgd",
        choices=["adam", "sgd"],
        help="Optimizer for training.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="best_model.pth",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory to load the model checkpoint from (defaults to output_dir_path).",
    )


def timing_parser(parser):
    configure_global_arguments(parser)

    subparsers = parser.add_subparsers(dest="dataset")
    mnistadd.configure_subparsers(subparsers)
    mnisthalf.configure_subparsers(subparsers)
    mnistsump.configure_subparsers(subparsers)
    mnistaddn.configure_subparsers(subparsers)
    derma.configure_subparsers(subparsers)
    chx.configure_subparsers(subparsers)
    boia.configure_subparsers(subparsers)
    cifar.configure_subparsers(subparsers)
    rival.configure_subparsers(subparsers)
    cebab.configure_subparsers(subparsers)
    mnistevenodd.configure_subparsers(subparsers)


def configure_subparsers(subparsers):
    """Configure the subparsers."""
    parser = subparsers.add_parser(
        "timing",
        help="Measure per-example inference latency for each conformal method",
    )
    timing_parser(parser)
    parser.set_defaults(func=main)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_baseline(model, timing_dl, device):
    """Time model.forward() over the full dataset.

    Warmup: one example (sufficient to warm GPU kernels for a simple
    encoder+inference pass).  Timing: one full pass over timing_dl synced
    before/after and divided by n_examples — the same "total / n" approach
    used by time_method, so both avoid the N extra cuda.synchronize() calls
    that per-example timing would introduce.

    Returns mean time in milliseconds per example.
    """
    model.eval()
    n = len(timing_dl.dataset)

    # Warmup: one example
    with torch.no_grad():
        data, _, _ = next(iter(timing_dl))
        model(data.to(device), eval=True)

    _sync(device)
    t0 = _time.perf_counter_ns()
    with torch.no_grad():
        for data, _, _ in timing_dl:
            model(data.to(device), eval=True)
    _sync(device)
    t1 = _time.perf_counter_ns()

    mean_ms = (t1 - t0) / 1e6 / n
    log(f"[Timing] No Conformal: {mean_ms:.4f} ms/example (n={n})", "INFO")
    return mean_ms


def time_method(predict_fn, device, n_examples, calibrate_fn=None):
    """Time a full predict_* call over the test set, divide by n_examples.

    calibrate_fn: runs calibration on val_dl — called once, NOT timed.
    predict_fn: callable that runs the full prediction loop over timing_dl.
    Returns mean time in milliseconds per example.
    """
    if calibrate_fn is not None:
        calibrate_fn()

    # Warmup: one untimed call (DataLoader resets iterator automatically)
    with torch.no_grad():
        predict_fn()

    _sync(device)
    t0 = _time.perf_counter_ns()
    with torch.no_grad():
        predict_fn()
    _sync(device)
    t1 = _time.perf_counter_ns()

    mean_ms = (t1 - t0) / 1e6 / n_examples
    return mean_ms


def timing_evaluation(model, val_dl, timing_dl, device, args, logic, experiment_name):
    """Measure per-example latency for all 10 conformal methods.

    Returns a list of dicts with keys: method, mean_time_ms, relative_to_baseline_pct.
    """
    multiconcept = args.dataset in ["boia", "chx", "derma", "cifar", "rival"]
    multilabel = args.dataset in ["boia"]
    n_examples = len(timing_dl.dataset)

    cp = ConformalPredictor(
        model,
        device=device,
        logic=logic,
        dataset=args.dataset,
        concept_dim=model.concept_dim,
        n_concepts=model.n_images,
        experiment_name=str(args.output_dir_path / experiment_name),
        multiconcepts=multiconcept,
        multilabel=multilabel,
        bonferroni=True,
    )

    cp_e = ConformalEPredictor(
        model,
        device=device,
        logic=logic,
        dataset=args.dataset,
        concept_dim=model.concept_dim,
        n_concepts=model.n_images,
        experiment_name=str(args.output_dir_path / experiment_name),
        multiconcepts=multiconcept,
        multilabel=multilabel,
    )

    # --- Calibrate ALL predictors once upfront, before any timed run ---
    # This ensures every method starts from the same GPU state and avoids
    # the cumulative warm-up bias that comes from re-calibrating inside each
    # time_method() call.
    log("=== Calibrating all predictors (untimed) ===", "INFO")
    cp.calibrate_per_concept(val_dl, alpha=0.1)
    cp.calibrate_labels(val_dl, alpha=0.1)
    cp_e.calibrate_per_concept(val_dl)
    cp_e.calibrate_labels(val_dl)

    results = []

    # --- 1. No Conformal (baseline) ---
    log("=== Timing 1. No Conformal ===", "INFO")
    baseline_ms = time_baseline(model, timing_dl, device)
    baseline_ms = max(baseline_ms, 1e-9)
    results.append({"method": "No Conformal", "mean_time_ms": baseline_ms})

    # --- 2. Conformal Concepts Only ---
    log("=== Timing 2. Conformal Concepts Only ===", "INFO")
    mean_ms = time_method(
        predict_fn=lambda: cp.predict_concepts(timing_dl),
        device=device,
        n_examples=n_examples,
    )
    log(f"[Timing] Conformal Concepts Only: {mean_ms:.4f} ms/example", "INFO")
    results.append({"method": "Conformal Concepts Only", "mean_time_ms": mean_ms})

    # --- 3. Conformal only Labels ---
    log("=== Timing 3. Conformal only Labels ===", "INFO")
    mean_ms = time_method(
        predict_fn=lambda: cp.predict_labels(timing_dl, use_hard_logic=False),
        device=device,
        n_examples=n_examples,
    )
    log(f"[Timing] Conformal only Labels: {mean_ms:.4f} ms/example", "INFO")
    results.append({"method": "Conformal only Labels", "mean_time_ms": mean_ms})

    # --- 4–9. predict_concepts_and_labels variants ---
    variants = [
        (
            "Conformal both Concepts and Labels",
            dict(use_hard_logic=False, concept_refinement=False, label_refinement=False),
        ),
        (
            "Conformal Hard Logic",
            dict(use_hard_logic=True, concept_refinement=False, label_refinement=False),
        ),
        (
            "Conformal with Abduction",
            dict(
                use_hard_logic=False,
                concept_refinement=False,
                label_refinement=False,
                abduction=True,
            ),
        ),
        (
            "Conformal with Concept Refinement",
            dict(use_hard_logic=False, concept_refinement=True, label_refinement=False),
        ),
        (
            "Conformal with Label Refinement",
            dict(use_hard_logic=False, concept_refinement=False, label_refinement=True),
        ),
        (
            "Conformal with Concept and Label Refinement",
            dict(use_hard_logic=False, concept_refinement=True, label_refinement=True),
        ),
    ]

    for method_name, kwargs in variants:
        log(f"=== Timing: {method_name} ===", "INFO")
        mean_ms = time_method(
            predict_fn=lambda kw=kwargs: cp.predict_concepts_and_labels(
                timing_dl, **kw
            ),
            device=device,
            n_examples=n_examples,
        )
        log(f"[Timing] {method_name}: {mean_ms:.4f} ms/example", "INFO")
        results.append({"method": method_name, "mean_time_ms": mean_ms})

    # --- 10. E-Value Refinement ---
    log("=== Timing 10. E-Value Refinement ===", "INFO")
    mean_ms = time_method(
        predict_fn=lambda: cp_e.predict_concepts_and_labels(
            timing_dl, alpha_labels=0.1, beta_concepts=0.1
        ),
        device=device,
        n_examples=n_examples,
    )
    log(f"[Timing] E-Value Refinement: {mean_ms:.4f} ms/example", "INFO")
    results.append({"method": "E-Value Refinement", "mean_time_ms": mean_ms})

    for r in results:
        r["relative_to_baseline_pct"] = (r["mean_time_ms"] / baseline_ms) * 100.0

    return results


def main(experiment_name, results_output_h, stats_output_h, args, device):
    """Main function: load model, run timing, write CSV."""

    (
        train_ds,
        val_ds,
        test_ds,
        input_dim,
        concept_dim,
        output_dim,
        n_images,
        class_names,
        concept_names,
        logic,
        criterion,
        concept_weights,
        label_weights,
    ) = DatasetFactory.get_dataset(args, name=args.dataset, device=args.device)

    n_test = len(test_ds)
    n_calib = int(0.2 * n_test)
    n_eval = n_test - n_calib
    calib_ds, eval_ds = random_split(
        test_ds, [n_calib, n_eval],
        generator=torch.Generator().manual_seed(args.seed),
    )
    _, calib_dl, _ = create_dataloaders(
        train_ds, calib_ds, eval_ds, batch_size=args.batch_size, shuffle_val=False
    )

    # batch_size=1 for per-example latency; num_workers=0 to exclude data-loading overhead
    timing_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)

    log("Loading the model...", "INFO")
    model = NetworkFactory.get_network(args.model, input_dim, concept_dim, args, n_images)
    model = NeSyFactory.get_nesy_model(
        args.nesy, n_images, model, concept_dim, output_dim, device, logic, args
    )

    if args.model_dir is not None:
        checkpoint_dir = args.model_dir
        saved_output_dir_path = args.output_dir_path
        args.output_dir_path = checkpoint_dir
        checkpoint_experiment_name = get_basename(args)
        args.output_dir_path = saved_output_dir_path
    else:
        checkpoint_dir = args.output_dir_path
        checkpoint_experiment_name = experiment_name
    model_path = checkpoint_dir / f"{checkpoint_experiment_name}.{args.model_path}.pth"
    model = load_model(model, model_path, device)
    model.to(device)

    logic_from_model = LogicFactory.get_logic(args.nesy, logic, model)

    results = timing_evaluation(
        model, calib_dl, timing_dl, device, args, logic_from_model, experiment_name
    )

    csv_path = args.output_dir_path / f"{experiment_name}.time_results.csv"
    log(f"Writing timing results to {csv_path}...", "INFO")

    fieldnames = [
        "seed",
        "dataset",
        "nesy",
        "concept_sup",
        "method",
        "mean_time_ms",
        "relative_to_baseline_pct",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "seed": args.seed,
                    "dataset": args.dataset,
                    "nesy": args.nesy,
                    "concept_sup": args.concept_supervision,
                    **r,
                }
            )

    log("Timing results written successfully.", "INFO")
