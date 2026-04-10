"""Test the Conformal Sets on the datasets."""

import numpy as np
import csv
import math
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler
from joblib import Parallel, delayed

from conformal.general_utils import log
from conformal.utils.factories import (
    DatasetFactory,
    NetworkFactory,
    NeSyFactory,
    LogicFactory,
)
from conformal.datasets.loaders import create_dataloaders
from conformal.experiments.utils import load_model, collect_predictions
from conformal.statistics.metrics import (
    conformal_metrics,
    prediction_consistency,
)
from conformal.models.econformal import ConformalEPredictor
from conformal.experiments.test import save_visual_examples
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
    """Configure global arguments that are shared across models and datasets."""

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


def test_parser(parser):
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
        "e-test",
        help="Evaluate conformal predictions on a dataset with a trained model",
    )
    test_parser(parser)
    parser.set_defaults(func=main)


def bootstrap_dataloader(dataset, batch_size=64, num_workers=0):
    sample_size = len(dataset)

    sampler = RandomSampler(dataset, replacement=True, num_samples=sample_size)

    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )


def avg_set_size(s):
    return np.mean([len(x) for x in s])


def find_minimum_set_per_size(conformal_sets, alphas, size):
    """Find the minimum alpha that achieves a conformal set of size at most `size`."""
    if not any(avg_set_size(s) <= size for s in conformal_sets):
        log("No alpha achieves the desired size", "INFO")
        log(
            f"Sizes of conformal sets: {[avg_set_size(s) for s in conformal_sets]}, Desired size: {size}",
            "INFO",
        )
        return None, None

    min_index = next(
        (i for i, x in enumerate(conformal_sets) if avg_set_size(x) <= size), -1
    )
    alpha_min = alphas[min_index]
    return alpha_min, min_index


def find_minimum_set_per_size_joint(
    label_sets, concept_sets, alphas, betas, max_label_size, max_concept_size
):
    """Find the minimum alpha and beta that achieves a conformal set of size at most `size` for both labels and concepts."""
    if not any(
        avg_set_size(l) <= max_label_size and avg_set_size(c) <= max_concept_size
        for l, c in zip(label_sets, concept_sets)
    ):
        log("No alpha and beta achieves the desired size", "INFO")
        log(
            f"Sizes of label sets: {[avg_set_size(l) for l in label_sets]}, Sizes of concept sets: {[avg_set_size(c) for c in concept_sets]}, Desired label size: {max_label_size}, Desired concept size: {max_concept_size}",
            "INFO",
        )
        return None, None, None

    min_index = next(
        (
            i
            for i, (l, c) in enumerate(zip(label_sets, concept_sets))
            if avg_set_size(l) <= max_label_size and avg_set_size(c) <= max_concept_size
        ),
        -1,
    )

    alpha_min = alphas[min_index // len(betas)]
    beta_min = betas[min_index % len(betas)]
    return alpha_min, beta_min, min_index


def get_conformal_metrics_for_size_fixed(
    args,
    is_image,
    test_dl,
    label_sets,
    concept_sets,
    all_labels,
    all_g,
    logic,
    label_min_idx,
    concept_min_idx,
):
    """
    Compute the conformal metrics for the setting where the size of the label set is fixed to a certain value.
    """
    concept_consistency, label_consistency = prediction_consistency(
        concept_sets[concept_min_idx], label_sets[label_min_idx], logic
    )

    concept_coverage, concept_set_size = conformal_metrics(
        concept_sets[concept_min_idx], all_g
    )

    label_coverage, label_size = conformal_metrics(
        label_sets[label_min_idx], np.expand_dims(all_labels, axis=1)
    )

    # save_visual_examples(
    #     test_dl.dataset,
    #     concept_sets[concept_min_idx],
    #     label_sets[label_min_idx],
    #     "Conformal both Concepts and Labels with Concept and Label Refinement",
    #     args.output_dir_path,
    #     is_image=is_image,
    # )

    return {
        "concept_consistency": concept_consistency,
        "label_consistency": label_consistency,
        "concept_coverage": concept_coverage,
        "concept_set_size": concept_set_size,
        "label_coverage": label_coverage,
        "label_set_size": label_size,
    }


def plot_one_minus_alpha(
    alpha_mins, alphas, output_path, prob, title="Histogram of $1-\\tilde{\\alpha}$"
):
    """
    Plot a histogram of 1 - alpha_mins and save the figure.

    Args:
        alpha_mins (list or np.array): list of minimum alpha values across iterations
        alphas (list or np.array): grid of alphas used in the evaluation
        output_path (str or Path): path to save the figure (PDF/PNG)
        title (str): title of the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    alpha_mins = np.array(alpha_mins)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram
    ax.hist(
        1 - alpha_mins,
        color="green",
        edgecolor="black",
        alpha=0.6,
        align="left",
    )

    # Mean line
    mean_val = np.mean(1 - alpha_mins)
    ax.axvline(
        x=mean_val,
        color="black",
        linestyle="--",
        linewidth=3,
        label=f"Mean: {mean_val:.3f}",
    )
    ax.axvline(x=prob, color="red", linestyle="--", linewidth=3)

    ax.set_xlabel(r"$1-\tilde{\alpha}$", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0.55, 1)
    ax.set_ylim(0, max(10, len(alpha_mins) // 2))  # adaptive y-axis

    ax.tick_params(axis="both", labelsize=14)
    ax.set_xticks(np.arange(0.1, 1.1, 0.1))
    ax.set_xticks(np.arange(0.05, 1.1, 0.05), minor=True)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Figure saved to {output_path}")


def median_set(sets):
    """Pick the set whose avg size is closest to the median avg size."""
    sizes = [avg_set_size(s) for s in sets]
    median_val = np.median(sizes)
    closest_idx = min(range(len(sizes)), key=lambda i: abs(sizes[i] - median_val))
    return sets[closest_idx]


def _run_single_iteration(
    iter_idx,
    model,
    device,
    experiment_name,
    multiconcept,
    multilabel,
    val_ds,
    test_ds,
    alphas,
    betas,
    all_labels,
    all_g,
    logic,
    args,
    is_image,
    expected_alpha,
    expected_beta,
    max_label_size,
    max_concept_size,
    n_iterations,
):
    """Run a single bootstrap iteration. Designed to be joblib-safe."""
    model.to(device)

    # Conformal EPredictor
    cp = ConformalEPredictor(
        model,
        device=device,
        logic=logic,
        dataset=args.dataset,
        concept_dim=model.concept_dim,
        n_concepts=model.n_images,
        experiment_name=str(args.output_dir_path / f"{experiment_name}"),
        multiconcepts=multiconcept,
        multilabel=multilabel,
    )

    import logging
    import sys

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        force=True,  # re-configures logging in the worker process
        format="%(message)s",
    )

    log(f"=== Iteration {iter_idx+1}/{n_iterations} ===", "INFO")

    # Each worker gets its own bootstrapped dataloader
    val_dl = bootstrap_dataloader(val_ds, batch_size=args.batch_size)
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Calibrate on this bootstrap sample
    cp.calibrate_per_concept(val_dl)
    cp.calibrate_labels(val_dl)

    label_sets_list, concept_sets_list = [], []
    label_sets_expected_alpha, concept_sets_expected_alpha = None, None

    for alpha in alphas:
        for beta in betas:
            c_sets, l_sets = cp.predict_concepts_and_labels(
                test_dl, alpha_labels=alpha, beta_concepts=beta
            )
            concept_coverage, concept_set_size = conformal_metrics(c_sets, all_g)
            label_coverage, label_set_size = conformal_metrics(
                l_sets, np.expand_dims(all_labels, axis=1)
            )

            log(
                f"[Iter {iter_idx+1}] Alpha: {alpha:.2f}, Beta: {beta:.2f} => "
                f"Label set size: {label_set_size}, Concept set size: {concept_set_size}, "
                f"Concept Coverage: {concept_coverage:.2f}, Label Coverage: {label_coverage:.2f}",
                "INFO",
            )

            concept_sets_list.append(c_sets)
            label_sets_list.append(l_sets)

            if math.isclose(alpha, expected_alpha, abs_tol=0.001) and math.isclose(
                beta, expected_beta, abs_tol=0.001
            ):
                label_sets_expected_alpha = [l_sets]
                concept_sets_expected_alpha = [c_sets]

    # Guard: expected alpha/beta must have been found in the grid
    if label_sets_expected_alpha is None or concept_sets_expected_alpha is None:
        log(
            f"[Iter {iter_idx+1}] Expected alpha={expected_alpha}/beta={expected_beta} "
            "not in grid — skipping iteration.",
            "INFO",
        )
        return None

    # Reshape flat list (len = n_alphas * n_betas) into a 2D structure

    # label_sets_by_alpha[i] = all sets for alphas[i] across all betas
    label_sets_by_alpha = [
        label_sets_list[i * len(betas) : (i + 1) * len(betas)]
        for i in range(len(alphas))
    ]
    label_sets_by_beta = [
        [label_sets_list[i * len(betas) + j] for i in range(len(alphas))]
        for j in range(len(betas))
    ]
    concept_sets_by_alpha = [
        concept_sets_list[i * len(betas) : (i + 1) * len(betas)]
        for i in range(len(alphas))
    ]
    # concept_sets_by_beta[j] = all sets for betas[j] across all alphas
    concept_sets_by_beta = [
        [concept_sets_list[i * len(betas) + j] for i in range(len(alphas))]
        for j in range(len(betas))
    ]

    # For each alpha bin, pick the best (smallest avg) set across betas
    label_sets_per_alpha = [min(sets, key=avg_set_size) for sets in label_sets_by_alpha]
    label_sets_per_beta = [min(sets, key=avg_set_size) for sets in label_sets_by_beta]
    # For each beta bin, pick the best (smallest avg) set across alphas
    concept_sets_per_beta = [
        min(sets, key=avg_set_size) for sets in concept_sets_by_beta
    ]
    concept_sets_per_alpha = [
        min(sets, key=avg_set_size) for sets in concept_sets_by_alpha
    ]

    # Find minimum alpha/beta achieving target max sizes
    alpha_min, label_min_idx = find_minimum_set_per_size(
        label_sets_per_alpha, alphas, max_label_size
    )
    beta_min, concept_min_idx = find_minimum_set_per_size(
        concept_sets_per_beta, betas, max_concept_size
    )
    both_min_alpha, both_min_beta, both_min_index = find_minimum_set_per_size_joint(
        label_sets_list,
        concept_sets_list,
        alphas,
        betas,
        max_label_size,
        max_concept_size,
    )

    if None in (
        alpha_min,
        label_min_idx,
        beta_min,
        concept_min_idx,
        both_min_alpha,
        both_min_beta,
        both_min_index,
    ):
        log(
            f"[Iter {iter_idx+1}] Could not find suitable alpha/beta, skipping.",
            "INFO",
        )
        return None

    metrics_fixed_labels = get_conformal_metrics_for_size_fixed(
        args,
        is_image,
        test_dl,
        label_sets_per_alpha,
        concept_sets_per_alpha,
        all_labels,
        all_g,
        logic,
        label_min_idx,
        label_min_idx,
    )
    metrics_fixed_worlds = get_conformal_metrics_for_size_fixed(
        args,
        is_image,
        test_dl,
        label_sets_per_beta,
        concept_sets_per_beta,
        all_labels,
        all_g,
        logic,
        concept_min_idx,
        concept_min_idx,
    )
    metrics_fixed_both = get_conformal_metrics_for_size_fixed(
        args,
        is_image,
        test_dl,
        label_sets_list,
        concept_sets_list,
        all_labels,
        all_g,
        logic,
        both_min_index,
        both_min_index,
    )
    metrics_expected = get_conformal_metrics_for_size_fixed(
        args,
        is_image,
        test_dl,
        label_sets_expected_alpha,
        concept_sets_expected_alpha,
        all_labels,
        all_g,
        logic,
        0,
        0,
    )

    return {
        "alpha_min": alpha_min,
        "beta_min": beta_min,
        "alpha_beta_min": (both_min_alpha, both_min_beta),
        "metrics_fixed_labels": metrics_fixed_labels,
        "metrics_fixed_worlds": metrics_fixed_worlds,
        "metrics_fixed_both": metrics_fixed_both,
        "metrics_expected": metrics_expected,
    }


def conformal_e_evaluation(
    model,
    val_dl,
    test_dl,
    device,
    args,
    logic,
    criterion,
    concept_names,
    experiment_name,
    val_ds,
    test_ds,
    alpha_concepts=0.1,
    alpha_label=0.1,
    n_iterations=100,
):
    log("Starting conformal prediction evaluation...", "INFO")

    log("Preparing the conformal e-predictor...", "INFO")

    multiconcept = (
        False
        if args.dataset not in ["boia", "chx", "derma", "cifar", "rival"]
        else True
    )
    multilabel = False if args.dataset not in ["boia"] else True
    is_image = args.dataset in ["boia", "chx", "derma", "cifar", "rival"]

    # log("Computing the permutation if needed...", "INFO")
    permutation = None

    log("Extracting the basic predictions...", "INFO")
    all_labels, _, all_g, _, _, _, _ = collect_predictions(
        model,
        args.dataset,
        test_dl,
        device,
        args.nesy,
        multiclass=True,  # To get the separated G
        multilabel=multilabel,
        permutation=permutation,
        is_dpl=(args.nesy == "dpl"),
    )

    # log("Computing the permutation if needed...", "INFO")
    permutation = None

    results_storage = {}

    alphas = np.arange(0.1, 0.3, 0.01)
    alphas = np.sort(alphas)

    betas = np.arange(0.3, 0.6, 0.01)
    betas = np.append(betas, 0.1)
    betas = np.sort(betas)

    max_concept_size = 5
    max_label_size = 2

    alpha_mins, beta_mins, alpha_beta_mins = [], [], []
    expected_alpha, expected_beta = 0.1, 0.1

    # --- Parallel loop over iterations ---
    model.cpu()
    iter_results = Parallel(n_jobs=3, backend="loky", verbose=10)(
        delayed(_run_single_iteration)(
            iter_idx=i,
            model=model,
            device=device,
            experiment_name=experiment_name,
            multiconcept=multiconcept,
            multilabel=multilabel,
            val_ds=val_ds,
            test_ds=test_ds,
            alphas=alphas,
            betas=betas,
            all_labels=all_labels,
            all_g=all_g,
            logic=logic,
            args=args,
            is_image=is_image,
            expected_alpha=expected_alpha,
            expected_beta=expected_beta,
            max_label_size=max_label_size,
            max_concept_size=max_concept_size,
            n_iterations=n_iterations,
        )
        for i in range(n_iterations)
    )
    model.to(device)

    # Collect results, skipping failed iterations
    for result in iter_results:
        if result is None:
            continue
        alpha_mins.append(result["alpha_min"])
        beta_mins.append(result["beta_min"])
        alpha_beta_mins.append(result["alpha_beta_min"])
        results_storage.setdefault("size_fixed_labels", []).append(
            result["metrics_fixed_labels"]
        )
        results_storage.setdefault("size_fixed_worlds", []).append(
            result["metrics_fixed_worlds"]
        )
        results_storage.setdefault("size_fixed_both", []).append(
            result["metrics_fixed_both"]
        )
        results_storage.setdefault("1-alpha-beta", []).append(
            result["metrics_expected"]
        )

    log("=== 1. Size fixed for the labels ===", "INFO")

    alpha_mean = np.mean(alpha_mins)
    one_minus_alpha_mean = 1 - alpha_mean
    results_storage["one_minus_alpha_mean"] = one_minus_alpha_mean
    log(f"Mean 1 - alpha across iterations: {one_minus_alpha_mean:.4f}", "INFO")

    prob_labels = np.mean(
        [m["label_coverage"] for m in results_storage["size_fixed_labels"]]
    )

    plot_one_minus_alpha(
        alpha_mins=alpha_mins,
        alphas=alphas,
        prob=prob_labels,
        title=f"Histogram of $1-\\tilde{{\\alpha}}$ ($Y \\leq {max_label_size}$, $T={n_iterations}$)",
        output_path=args.output_dir_path
        / f"{experiment_name}_1_minus_alpha_hist_full.pdf",
    )

    log("=== 2. Size fixed for the worlds ===", "INFO")

    beta_mean = np.mean(beta_mins)
    one_minus_beta_mean = 1 - beta_mean
    results_storage["one_minus_beta_mean"] = one_minus_beta_mean
    log(f"Mean 1 - beta across iterations: {one_minus_beta_mean:.4f}", "INFO")

    prob_worlds = np.mean(
        [m["concept_coverage"] for m in results_storage["size_fixed_worlds"]]
    )

    plot_one_minus_alpha(
        alpha_mins=beta_mins,
        alphas=betas,
        prob=prob_worlds,
        title=f"Histogram of $1-\\tilde{{\\beta}}$ ($C \\leq {max_concept_size}$, $T={n_iterations}$)",
        output_path=args.output_dir_path
        / f"{experiment_name}_1_minus_beta_hist_full.pdf",
    )

    log("=== 3. Size fixed for both labels and worlds ===", "INFO")

    alpha_both_mean = np.mean([alpha for alpha, _ in alpha_beta_mins])
    one_minus_alpha_both_mean = 1 - alpha_both_mean
    results_storage["one_minus_alpha_both_mean"] = one_minus_alpha_both_mean
    log(
        f"Mean 1 - alpha (both) across iterations: {one_minus_alpha_both_mean:.4f}",
        "INFO",
    )

    beta_both_mean = np.mean([beta for _, beta in alpha_beta_mins])
    one_minus_beta_both_mean = 1 - beta_both_mean
    results_storage["one_minus_beta_both_mean"] = one_minus_beta_both_mean
    log(
        f"Mean 1 - beta (both) across iterations: {one_minus_beta_both_mean:.4f}",
        "INFO",
    )

    prob_both_alpha = np.mean(
        [m["label_coverage"] for m in results_storage["size_fixed_both"]]
    )
    prob_both_beta = np.mean(
        [m["concept_coverage"] for m in results_storage["size_fixed_both"]]
    )

    plot_one_minus_alpha(
        alpha_mins=[alpha for alpha, _ in alpha_beta_mins],
        alphas=alphas,
        prob=prob_both_alpha,
        title=f"Histogram of $1-\\tilde{{\\alpha}}$ ($Y \\leq {max_label_size}$, $C \\leq {max_concept_size}$, $T={n_iterations}$)",
        output_path=args.output_dir_path
        / f"{experiment_name}_1_minus_alpha_both_hist_full.pdf",
    )

    plot_one_minus_alpha(
        alpha_mins=[beta for _, beta in alpha_beta_mins],
        alphas=betas,
        prob=prob_both_beta,
        title=f"Histogram of $1-\\tilde{{\\beta}}$ ($Y \\leq {max_label_size}$, $C \\leq {max_concept_size}$, $T={n_iterations}$)",
        output_path=args.output_dir_path
        / f"{experiment_name}_1_minus_beta_both_hist_full.pdf",
    )

    log("Aggregating results across iterations for each setting...", "INFO")

    log("=== 4. Joint Conformality with 1 - E[alpha] - E[beta] ===", "INFO")

    for key in [
        "size_fixed_labels",
        "size_fixed_worlds",
        "size_fixed_both",
        "1-alpha-beta",
    ]:
        aggregated = {}
        for metric_name in results_storage[key][0].keys():
            aggregated[metric_name] = np.mean(
                [d[metric_name] for d in results_storage[key]], axis=0
            )
        results_storage[f"aggregated_{key}"] = aggregated

    log("=== 5. Save everything ===", "INFO")

    # === 1. Size fixed for the labels ===
    prob_labels = [m["label_coverage"] for m in results_storage["size_fixed_labels"]]
    label_size = [m["label_set_size"] for m in results_storage["size_fixed_labels"]]
    concept_size = [m["concept_set_size"] for m in results_storage["size_fixed_labels"]]
    csv_path = args.output_dir_path / f"{experiment_name}_1_minus_alpha_hist_full.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "file_name",
                "c",
                "t",
                "y",
                "1_minus_alpha",
                "prob",
                "alpha_mean",
                "1_minus_alpha_mean",
                "alphas_grid",
                "betas_grid",
                "label_size",
                "concept_size",
            ]
        )
        for am, a, b, p, l, c in zip(
            alpha_mins, alphas, betas, prob_labels, label_size, concept_size
        ):
            writer.writerow(
                [
                    experiment_name,
                    f"{experiment_name}_1_minus_alpha_hist_full.pdf",
                    max_concept_size,
                    n_iterations,
                    max_label_size,
                    1 - am,
                    p,
                    round(alpha_mean, 6),
                    round(one_minus_alpha_mean, 6),
                    a,
                    b,
                    l,
                    c,
                ]
            )

    # === 2. Size fixed for the worlds ===
    prob_worlds = [m["concept_coverage"] for m in results_storage["size_fixed_worlds"]]
    label_size = [m["label_set_size"] for m in results_storage["size_fixed_worlds"]]
    concept_size = [m["concept_set_size"] for m in results_storage["size_fixed_worlds"]]
    csv_path = args.output_dir_path / f"{experiment_name}_1_minus_beta_full.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "file_name",
                "c",
                "t",
                "y",
                "1_minus_beta",
                "prob",
                "beta_mean",
                "1_minus_beta_mean",
                "alphas_grid",
                "betas_grid",
                "label_size",
                "concept_size",
            ]
        )
        for bm, b, a, p, l, c in zip(
            beta_mins, betas, alphas, prob_worlds, label_size, concept_size
        ):
            writer.writerow(
                [
                    experiment_name,
                    f"{experiment_name}_1_minus_beta_hist_full.pdf",
                    max_concept_size,
                    n_iterations,
                    max_label_size,
                    1 - bm,
                    p,
                    round(beta_mean, 6),
                    round(one_minus_beta_mean, 6),
                    a,
                    b,
                    l,
                    c,
                ]
            )

    # === 3. Size fixed for both — alpha side ===
    label_sizes = [m["label_set_size"] for m in results_storage["size_fixed_both"]]
    concept_sizes = [m["concept_set_size"] for m in results_storage["size_fixed_both"]]
    csv_path = args.output_dir_path / f"{experiment_name}_1_minus_alpha_both_full.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "name",
                "file_name",
                "c",
                "t",
                "y",
                # shared grids
                "alphas_grid",
                "betas_grid",
                # alpha side
                "1_minus_alpha",
                "alpha_prob",
                "alpha_mean",
                "1_minus_alpha_mean",
                # beta side
                "1_minus_beta",
                "beta_prob",
                "beta_mean",
                "1_minus_beta_mean",
                # sizes
                "label_size",
                "concept_size",
            ]
        )

        for (alpha, beta), a, b, p_alpha, p_beta, l, c in zip(
            alpha_beta_mins,
            alphas,
            betas,
            [m["label_coverage"] for m in results_storage["size_fixed_both"]],
            [m["concept_coverage"] for m in results_storage["size_fixed_both"]],
            label_sizes,
            concept_sizes,
        ):
            writer.writerow(
                [
                    experiment_name,
                    f"{experiment_name}_1_minus_both_hist_full.pdf",
                    max_concept_size,
                    n_iterations,
                    max_label_size,
                    a,
                    b,
                    1 - alpha,
                    p_alpha,
                    round(alpha_both_mean, 6),
                    round(one_minus_alpha_both_mean, 6),
                    1 - beta,
                    p_beta,
                    round(beta_both_mean, 6),
                    round(one_minus_beta_both_mean, 6),
                    l,
                    c,
                ]
            )

    return results_storage


def main(experiment_name, results_output_h, stats_output_h, args, device):
    """Main function that parses the arguments and writes the output."""

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

    _, val_dl, test_dl = create_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=args.batch_size,
        shuffle_val=False,
        num_workers=0,
    )

    log("Loading the model", "INFO")

    model = NetworkFactory.get_network(
        args.model, input_dim, concept_dim, args, n_images
    )
    model = NeSyFactory.get_nesy_model(
        args.nesy, n_images, model, concept_dim, output_dim, device, logic, args
    )

    model_path = args.output_dir_path / f"{experiment_name}.{args.model_path}.pth"
    model = load_model(model, model_path, device)
    model.to(device)

    # load the logic
    logic_from_model = LogicFactory.get_logic(args.nesy, logic, model)

    alpha_concepts = 0.1
    alpha_label = 0.1
    n_iterations = 50

    log("Adding e-value as folder")
    args.output_dir_path = args.output_dir_path / "evalues"
    args.output_dir_path.mkdir(parents=True, exist_ok=True)

    result_storage = conformal_e_evaluation(
        model,
        val_dl,
        test_dl,
        device,
        args,
        logic_from_model,
        criterion,
        concept_names,
        experiment_name,
        alpha_concepts=alpha_concepts,
        alpha_label=alpha_label,
        n_iterations=n_iterations,
        val_ds=val_ds,
        test_ds=test_ds,
    )

    log("Results Summary:", "INFO")
    for setting, metrics in result_storage.items():
        log(f"Setting: {setting}", "INFO")
        log(f"  {metrics}", "INFO")

    original_path = Path(stats_output_h.name)
    conformal_stats_path = original_path.with_name(
        f"{original_path.stem}_e_conformal{original_path.suffix}"
    )

    log(f"Writing conformal results to {conformal_stats_path}...", "INFO")

    with open(conformal_stats_path, "w") as f:
        for setting, metrics in result_storage.items():
            f.write(f"# Results for setting: {setting}\n")
            if type(metrics) == list:
                for el in metrics:
                    for metric_name, metric_value in el.items():
                        if isinstance(metric_value, float):
                            f.write(f"{metric_name},{metric_value:.5f}\n")
                        else:
                            f.write(f"{metric_name},{metric_value}\n")
            elif type(metrics) == dict:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        f.write(f"{metric_name},{metric_value:.5f}\n")
                    else:
                        f.write(f"{metric_name},{metric_value}\n")
            else:
                if isinstance(metric_value, float):
                    f.write(f"{metric_name},{metric_value:.5f}\n")
                else:
                    f.write(f"{metric_name},{metric_value}\n")
            f.write("\n")

    log("Results written successfully.", "INFO")


if __name__ == "__main__":
    main()
