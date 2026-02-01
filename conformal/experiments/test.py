"""Test the Conformal Sets on the datasets."""

import numpy as np
import textwrap
from pathlib import Path
import matplotlib.pyplot as plt

from conformal.models import resnet18, lenet, linear
from conformal.general_utils import log
from conformal.utils.factories import (
    DatasetFactory,
    NetworkFactory,
    NeSyFactory,
    LogicFactory,
)
from conformal.datasets.loaders import create_dataloaders
from conformal.experiments.utils import load_model
from conformal.statistics.metrics import compute_statistics, conformal_metrics
from conformal.experiments.utils import collect_predictions
from conformal.models.conformal import ConformalPredictor
from conformal.utils.visualization import plot_conformal_comparison, plot_model_metrics


def configure_global_arguments(parser):
    """Configure global arguments that are shared across models and datasets."""

    parser.add_argument(
        "dataset",
        metavar="DATASET",
        choices={"mnistadd", "mnisthalf", "mnistsump", "boia", "chx", "derma"},
        default="mnistadd",
        help="Dataset",
    )
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

    subparsers = parser.add_subparsers(dest="model")
    resnet18.configure_subparsers(subparsers)
    lenet.configure_subparsers(subparsers)
    linear.configure_subparsers(subparsers)


def configure_subparsers(subparsers):
    """Configure the subparsers."""
    parser = subparsers.add_parser(
        "test",
        help="Evaluate conformal predictions on a dataset with a trained model",
    )
    test_parser(parser)
    parser.set_defaults(func=main)


def save_visual_examples(
    dataset, concept_sets, label_sets, method_name, output_dir, limit=10, is_image=True
):
    """
    Finds and saves samples where at least one concept prediction set has size >= 2.
    """
    clean_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = output_dir / "visual_examples" / clean_name
    save_path.mkdir(parents=True, exist_ok=True)

    count = 0

    for i in range(len(concept_sets)):
        if count >= limit:
            break

        c_sets = concept_sets[i].tolist()
        l_sets = label_sets[i].tolist() if label_sets is not None else None

        # CONDITION: Check if any concept has a set size >= 2 (Uncertainty)
        # If your concepts are binary, >=2 means the set is {0, 1} (Don't Know)
        is_interesting = any(len(s) != 1 for s in c_sets)

        if is_interesting:
            image_data, true_concepts, true_label = dataset[i]

            # Setup Plot
            fig = plt.figure(figsize=(8, 6))

            # Layout: Image on top, Text details below
            gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
            ax_img = fig.add_subplot(gs[0])
            ax_txt = fig.add_subplot(gs[1])

            if is_image:
                if hasattr(image_data, "permute"):  # PyTorch Tensor
                    img_np = image_data.permute(1, 2, 0).cpu().numpy()
                    # Simple un-normalization (clip to 0-1) for visualization
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                else:
                    img_np = np.array(image_data)

                # Show Image
                if img_np.shape[-1] == 1:  # Grayscale
                    ax_img.imshow(img_np, cmap="gray")
                else:
                    ax_img.imshow(img_np)

            ax_img.axis("off")
            ax_img.set_title(
                f"Sample {i} | Method: {method_name}", fontsize=12, fontweight="bold"
            )

            # Clean up long arrays for display
            wrapped_text = textwrap.fill(str(c_sets), width=80)

            ax_txt.text(
                0,
                1,
                f"True Label: {true_label}\nTrue Concepts: {true_concepts}\n\n"
                f"Pred Label Set: {l_sets}\n"
                f"Pred Concept Sets:\n{wrapped_text}",
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
            ax_txt.axis("off")

            # Save
            fname = save_path / f"sample_{i}.pdf"
            log(fname, "DEBUG")
            plt.savefig(fname, format="pdf", bbox_inches="tight")
            plt.close()

            count += 1

    log(f"[Info] Saved {count} examples for {method_name} to {save_path}", "INFO")


def conformal_evaluation(
    model,
    val_dl,
    test_dl,
    device,
    args,
    logic,
    dataset,
    criterion,
    alpha_concepts=0.1,
    alpha_label=0.1,
):
    log("Starting conformal prediction evaluation...", "INFO")

    results_storage = {}

    multiconcept = False if args.dataset not in ["boia", "chx", "derma"] else True
    multilabel = False if args.dataset not in ["boia"] else True
    is_image = False if args.dataset in ["boia"] else True

    log("=== 1. Baseline (Standard Argmax) ===", "INFO")

    (
        test_loss,
        test_f1,
        test_c_f1,
        test_H_c,
        test_H_c_per_value,
        test_yece,
        test_cece,
        _,
    ) = compute_statistics(
        model,
        args.dataset,
        test_dl,
        criterion,
        device,
        multiclass=multiconcept,
        multilabel=multilabel,
    )

    all_labels, all_preds, all_g, _, _, _, _ = collect_predictions(
        model,
        test_dl,
        device,
        multiclass=True,  # To get the separated G
        multilabel=multilabel,
    )

    results_storage["No Conformal"] = {
        "test_loss": test_loss,
        "test_f1": test_f1,
        "test_c_f1": test_c_f1,
        "H_c": test_H_c,
        "H_c_per_value": test_H_c_per_value,
        "yece": test_yece,
        "cece": test_cece,
    }

    log("Preparing the conformal predictor...", "INFO")

    cp = ConformalPredictor(
        model,
        device=device,
        logic=logic,
        dataset=dataset,
        concept_dim=model.concept_dim,
        n_concepts=model.n_images,
        multiconcepts=multiconcept,
        multilabel=multilabel,
        bonferroni=(True if args.dataset in ["chx", "derma"] else False),
    )

    log("=== 2. Conformal (Calibrating Concepts) ===", "INFO")

    log("Calibrating concept thresholds...", "INFO")
    cp.calibrate_per_concept(val_dl, alpha=alpha_concepts)

    log("Predicting conformal sets on the test set...", "INFO")
    concept_sets = cp.predict_concepts(test_dl)

    concept_coverage, concept_set_size = conformal_metrics(concept_sets, all_g)

    log(f"[Conformal Concepts Only] Concept Coverage: {concept_coverage:.4f}", "INFO")
    log(f"[Conformal Concepts Only] Concept Set Size: {concept_set_size:.4f}", "INFO")

    results_storage["Conformal Concepts Only"] = {
        "coverage_concepts": concept_coverage,
        "concept_size": concept_set_size,
    }

    save_visual_examples(
        test_dl.dataset,
        concept_sets,
        all_preds,
        "Conformal Concepts Only",
        args.output_dir_path,
        is_image=is_image,
    )

    log("=== 3. Conformal (Calibrating both concepts and labels) ===", "INFO")

    log("Calibrating label thresholds...", "INFO")
    cp.calibrate_labels(val_dl, alpha=alpha_label)

    concept_sets, label_sets = cp.predict_concepts_and_labels(
        test_dl, use_hard_logic=False, concept_refinement=False
    )

    label_coverage, label_size = conformal_metrics(label_sets, all_labels)

    log(
        f"[Conformal both Concepts and Labels] Label Coverage: {label_coverage:.4f}",
        "INFO",
    )
    log(
        f"[Conformal both Concepts and Labels] Label Set Size: {label_size:.4f}", "INFO"
    )

    results_storage["Conformal both Concepts and Labels"] = {
        "coverage_concepts": concept_coverage,
        "concept_size": concept_set_size,
        "coverage_labels": label_coverage,
        "label_size": label_size,
    }

    save_visual_examples(
        test_dl.dataset,
        concept_sets,
        label_sets,
        "Conformal both Concepts and Labels",
        args.output_dir_path,
        is_image=is_image,
    )

    log("=== 4. Conformal (Hard Logic) ===", "INFO")

    concept_sets, label_sets = cp.predict_concepts_and_labels(
        test_dl, use_hard_logic=True
    )

    label_coverage, label_size = conformal_metrics(label_sets, all_labels)

    log(f"[Conformal Hard Logic] Label Coverage: {label_coverage:.4f}", "INFO")
    log(f"[Conformal Hard Logic] Label Set Size: {label_size:.4f}", "INFO")

    results_storage["Conformal Hard Logic"] = {
        "coverage_concepts": concept_coverage,
        "concept_size": concept_set_size,
        "coverage_labels": label_coverage,
        "label_size": label_size,
    }

    save_visual_examples(
        test_dl.dataset,
        concept_sets,
        label_sets,
        "Conformal Hard Logic",
        args.output_dir_path,
        is_image=is_image,
    )

    log("=== 5. Conformal with Concept Refinement ===", "INFO")

    concept_sets, label_sets = cp.predict_concepts_and_labels(
        test_dl, use_hard_logic=False, concept_refinement=True
    )

    concept_coverage, concept_set_size = conformal_metrics(concept_sets, all_g)
    label_coverage, label_size = conformal_metrics(label_sets, all_labels)

    log(
        f"[Conformal both Concepts and Labels with Concept Refinement] Concept Coverage: {concept_coverage:.4f}",
        "INFO",
    )
    log(
        f"[Conformal both Concepts and Labels with Concept Refinement] Concept Set Size: {concept_set_size:.4f}",
        "INFO",
    )
    log(
        f"[Conformal both Concepts and Labels with Concept Refinement] Label Coverage: {label_coverage:.4f}",
        "INFO",
    )
    log(
        f"[Conformal both Concepts and Labels with Concept Refinement] Label Set Size: {label_size:.4f}",
        "INFO",
    )

    save_visual_examples(
        test_dl.dataset,
        concept_sets,
        label_sets,
        "Conformal both Concepts and Labels with Concept Refinement",
        args.output_dir_path,
        is_image=is_image,
    )

    results_storage["Conformal both Concepts and Labels with Concept Refinement"] = {
        "coverage_concepts": concept_coverage,
        "concept_size": concept_set_size,
        "coverage_labels": label_coverage,
        "label_size": label_size,
    }

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
    ) = DatasetFactory.get_dataset(name=args.dataset, device=args.device)

    _, val_dl, test_dl = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size, shuffle_val=False
    )

    log("Loading the model", "INFO")

    model = NetworkFactory.get_network(args.model, input_dim, concept_dim, args)
    model = NeSyFactory.get_nesy_model(
        args.nesy, n_images, model, concept_dim, output_dim, device, logic, args
    )

    model_path = args.output_dir_path / f"{experiment_name}.{args.model_path}.pth"
    model = load_model(model, model_path, device)
    model.to(device)

    # load the logic
    logic_from_model = LogicFactory.get_logic(args.nesy, logic, model)

    alpha = 0.1
    result_storage = conformal_evaluation(
        model,
        val_dl,
        test_dl,
        device,
        args,
        logic_from_model,
        args.dataset,
        criterion,
        alpha_concepts=alpha,
        alpha_label=alpha,
    )

    plot_conformal_comparison(
        result_storage, args.output_dir_path, target_coverage=1 - alpha
    )

    plot_model_metrics(
        result_storage["No Conformal"],
        args.output_dir_path,
        concept_names=args.concept_names if hasattr(args, "concept_names") else None,
        multiconcepts=True if args.dataset in ["boia", "chx", "derma"] else False,
    )

    log("Results Summary:", "INFO")
    for setting, metrics in result_storage.items():
        log(f"Setting: {setting}", "INFO")
        for metric_name, metric_value in metrics.items():
            log(f"  {metric_name}: {metric_value}", "INFO")

    original_path = Path(stats_output_h.name)
    conformal_stats_path = original_path.with_name(
        f"{original_path.stem}_conformal{original_path.suffix}"
    )

    log(f"Writing conformal results to {conformal_stats_path}...", "INFO")

    with open(conformal_stats_path, "w") as f:
        for setting, metrics in result_storage.items():
            f.write(f"# Results for setting: {setting}\n")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    f.write(f"{metric_name},{metric_value:.5f}\n")
                else:
                    f.write(f"{metric_name},{metric_value}\n")
            f.write("\n")

    log("Results written successfully.", "INFO")


if __name__ == "__main__":
    main()
