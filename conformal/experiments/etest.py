"""Test the Conformal Sets on the datasets."""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, SubsetRandomSampler

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


def bootstrap_dataloader(dataset, batch_size=64, num_workers=2):
    """
    Return a DataLoader that performs bootstrap sampling on the dataset.

    Args:
        dataset (torch.utils.data.Dataset): input dataset
        batch_size (int): batch size for DataLoader
        num_workers (int): number of workers for DataLoader

    Returns:
        DataLoader
    """
    n = len(dataset)
    sampled_indices = torch.randint(0, n, (n,), dtype=torch.long)
    sampler = SubsetRandomSampler(sampled_indices)
    dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return dl


def find_minimum_set_per_size(conformal_sets, alphas, size):
    """Find the minimum alpha that achieves a conformal set of size at most `size`."""
    min_index = next((i for i, x in enumerate(conformal_sets) if len(x) <= size), -1)
    alpha_min = alphas[min_index]
    return alpha_min, min_index

def find_minimum_set_per_size_joint(label_sets, concept_sets, alphas, betas, max_label_size, max_concept_size):
    """Find the minimum alpha and beta that achieves a conformal set of size at most `size` for both labels and concepts."""
    min_index = next((i for i, (l, c) in enumerate(zip(label_sets, concept_sets)) if len(l) <= max_label_size and len(c) <= max_concept_size), -1)
    alpha_min = alphas[min_index // len(betas)]
    beta_min = betas[min_index % len(betas)]
    return alpha_min, beta_min, min_index


def get_conformal_metrics_for_size_fixed(args, is_image, test_dl, label_sets, concept_sets, all_labels, all_g, logic, label_min_idx, concept_min_idx):
    """
    Compute the conformal metrics for the setting where the size of the label set is fixed to a certain value.
    """
    concept_consistency, label_consistency = prediction_consistency(
        concept_sets[concept_min_idx], label_sets[label_min_idx], logic
    )

    concept_coverage, concept_set_size = conformal_metrics(concept_sets[concept_min_idx], all_g)

    label_coverage, label_size = conformal_metrics(
        label_sets[label_min_idx], np.expand_dims(all_labels, axis=1)
    )

    save_visual_examples(
        test_dl.dataset,
        concept_sets[concept_min_idx],
        label_sets[label_min_idx],
        "Conformal both Concepts and Labels with Concept and Label Refinement",
        args.output_dir_path,
        is_image=is_image,
    )

    return {
        "concept_consistency": concept_consistency,
        "label_consistency": label_consistency,
        "concept_coverage": concept_coverage,
        "concept_set_size": concept_set_size,
        "label_coverage": label_coverage,
        "label_set_size": label_size,
    }


def plot_one_minus_alpha(alpha_mins, alphas, output_path, title="Histogram of $1-\\tilde{\\alpha}$"):
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
        bins=1 - np.array(alphas[::-1]),
        align="left",
    )

    # Mean line
    mean_val = np.mean(1 - alpha_mins)
    ax.axvline(x=mean_val, color="black", linestyle="--", linewidth=3, label=f"Mean: {mean_val:.3f}")

    ax.set_xlabel(r"$1-\tilde{\alpha}$", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0.8, 1)
    ax.set_ylim(0, max(10, len(alpha_mins)//2))  # adaptive y-axis

    ax.tick_params(axis="both", labelsize=14)
    ax.set_xticks(np.linspace(0.8, 1.0, 11))
    ax.set_xticks(np.linspace(0.8, 1.0, 21), minor=True)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Figure saved to {output_path}")



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

    log("Computing the permutation if needed...", "INFO")
    permutation = None

    if args.concept_supervision == 0.0 and args.nesy not in ["dpl", "ltn"]:
        log("Computing the permutation matrix since concepts cannot be inferred...")
        permutation = cp.compute_permutation(val_dl)
        cp.set_permutation(permutation)

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
    )

    log("Computing the permutation if needed...", "INFO")
    permutation = None

    if args.concept_supervision == 0.0 and args.nesy not in ["dpl", "ltn"]:
        log("Computing the permutation matrix since concepts cannot be inferred...")
        permutation = cp.compute_permutation(val_dl)
        cp.set_permutation(permutation)


    results_storage = {}

    alphas = np.arange(0.01,0.03,0.01) # 0.31
    betas = np.arange(0.01,0.03,0.01) # 0.31
    max_concept_size = 3
    max_label_size = 2
    alpha_mins, beta_mins, alpha_beta_mins = [], [], []
    expected_alpha, expected_beta = 0.01, 0.01


    for iter in range(n_iterations):
        log(f"=== Iteration {iter+1}/{n_iterations} ===", "INFO")

        val_dl = bootstrap_dataloader(val_ds, batch_size=args.batch_size)

        # compute the calibration scores
        cp.calibrate_per_concept(val_dl)
        cp.calibrate_labels(val_dl)

        label_sets_list, concept_sets_list = [], []
        label_sets_expected_alpha, concept_sets_expected_alpha = None, None

        for alpha in alphas:
            for beta in betas:
                c_sets, l_sets = cp.predict_concepts_and_labels(
                    test_dl, alpha_labels=alpha, beta_concepts=beta
                )
                log(f"Alpha: {alpha:.2f}, Beta: {beta:.2f} => Label set size: {np.mean([len(s) for s in l_sets]):.2f}, Concept set size: {np.mean([len(s) for s in c_sets]):.2f}", "INFO")
                concept_sets_list.append(c_sets)
                label_sets_list.append(l_sets)
        
                # save only those
                if alpha == expected_alpha and beta == expected_beta:
                    label_sets_expected_alpha = [l_sets]
                    concept_sets_expected_alpha = [c_sets]
        
        # Find minimum alpha/beta achieving target max sizes
        alpha_min, label_min_idx = find_minimum_set_per_size(label_sets_list, alphas, max_label_size)
        beta_min, concept_min_idx = find_minimum_set_per_size(concept_sets_list, betas, max_concept_size)
        both_min_alpha, both_min_beta, both_min_index = find_minimum_set_per_size_joint(
            label_sets_list, concept_sets_list, alphas, betas, max_label_size, max_concept_size
        )

        alpha_mins.append(alpha_min)
        beta_mins.append(beta_min)
        alpha_beta_mins.append((both_min_alpha, both_min_beta))

        # Compute metrics for the three “size-fixed” settings
        metrics_fixed_labels = get_conformal_metrics_for_size_fixed(
            args, is_image, test_dl, label_sets_list, concept_sets_list, all_labels, all_g, logic, label_min_idx, label_min_idx
        )
        metrics_fixed_worlds = get_conformal_metrics_for_size_fixed(
            args, is_image, test_dl, label_sets_list, concept_sets_list, all_labels, all_g, logic, concept_min_idx, concept_min_idx
        )
        metrics_fixed_both = get_conformal_metrics_for_size_fixed(
            args, is_image, test_dl, label_sets_list, concept_sets_list, all_labels, all_g, logic, both_min_index, both_min_index
        )
        metrics_expected = get_conformal_metrics_for_size_fixed(
            args, is_image, test_dl, label_sets_expected_alpha, concept_sets_expected_alpha, all_labels, all_g, logic, 0, 0
        )

        # Store metrics for this iteration
        results_storage.setdefault("size_fixed_labels", []).append(metrics_fixed_labels)
        results_storage.setdefault("size_fixed_worlds", []).append(metrics_fixed_worlds)
        results_storage.setdefault("size_fixed_both", []).append(metrics_fixed_both)
        results_storage.setdefault("1-alpha-beta", []).append(metrics_expected)
    
    log("=== 1. Size fixed for the labels ===", "INFO")

    alpha_mean = np.mean(alpha_mins)
    one_minus_alpha_mean = 1 - alpha_mean
    results_storage["one_minus_alpha_mean"] = one_minus_alpha_mean
    log(f"Mean 1 - alpha across iterations: {one_minus_alpha_mean:.4f}", "INFO")

    plot_one_minus_alpha(
        alpha_mins=alpha_mins,
        alphas=alphas,
        output_path=args.output_dir_path / f"{experiment_name}_1_minus_alpha_hist.pdf"
    )

    log("=== 2. Size fixed for the worlds ===", "INFO")

    beta_mean = np.mean(beta_mins)
    one_minus_beta_mean = 1 - beta_mean
    results_storage["one_minus_beta_mean"] = one_minus_beta_mean
    log(f"Mean 1 - beta across iterations: {one_minus_beta_mean:.4f}", "INFO")

    plot_one_minus_alpha(
        alpha_mins=beta_mins,
        alphas=betas,
        output_path=args.output_dir_path / f"{experiment_name}_1_minus_beta_hist.pdf"
    )

    log("=== 3. Size fixed for both labels and worlds ===", "INFO")

    alpha_both_mean = np.mean([alpha for alpha, _ in alpha_beta_mins])
    one_minus_alpha_both_mean = 1 - alpha_both_mean
    results_storage["one_minus_alpha_both_mean"] = one_minus_alpha_both_mean
    log(f"Mean 1 - alpha (both) across iterations: {one_minus_alpha_both_mean:.4f}", "INFO")

    beta_both_mean = np.mean([beta for _, beta in alpha_beta_mins])
    one_minus_beta_both_mean = 1 - beta_both_mean
    results_storage["one_minus_beta_both_mean"] = one_minus_beta_both_mean
    log(f"Mean 1 - beta (both) across iterations: {one_minus_beta_both_mean:.4f}", "INFO")

    plot_one_minus_alpha(
        alpha_mins=[alpha for alpha, _ in alpha_beta_mins],
        alphas=alphas,
        output_path=args.output_dir_path / f"{experiment_name}_1_minus_alpha_both_hist.pdf"
    )

    plot_one_minus_alpha(
        alpha_mins=[beta for _, beta in alpha_beta_mins],
        alphas=betas,
        output_path=args.output_dir_path / f"{experiment_name}_1_minus_beta_both_hist.pdf"
    )

    log("Aggregating results across iterations for each setting...", "INFO")

    log("=== 4. Joint Conformality with 1 - E[alpha] - E[beta] ===", "INFO")

    for key in ["size_fixed_labels", "size_fixed_worlds", "size_fixed_both", "1-alpha-beta"]:
        aggregated = {}
        for metric_name in results_storage[key][0].keys():
            aggregated[metric_name] = np.mean([d[metric_name] for d in results_storage[key]], axis=0)
        results_storage[f"aggregated_{key}"] = aggregated

    log("=== 5. Joint Conformality with 1 - E[gamma] ===", "INFO")

    # TODO Kronecker?

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
        train_ds, val_ds, test_ds, batch_size=args.batch_size, shuffle_val=False
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
    n_iterations = 2

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
