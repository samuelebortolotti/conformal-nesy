"""Test a CBM model and track concept entropy and F1 metrics."""

import torch
import math
import random
import copy

import numpy as np

from conformal.models import resnet18, lenet, linear
from conformal.general_utils import log
from conformal.utils.factories import DatasetFactory, NetworkFactory, OptimizerFactory
from conformal.models.dpl import DPL
from conformal.datasets.loaders import create_dataloaders
from conformal.utils.visualization import (
    save_line_plot,
    plot_confusion_matrix,
    entropy_plot_over_time,
)
from conformal.experiments.train import compute_statistics, collect_predictions
from conformal.utils.other import outer_product


def configure_global_arguments(parser):
    """Configure global arguments that are shared across models and datasets."""

    parser.add_argument(
        "dataset",
        metavar="DATASET",
        choices={"mnistadd", "mnisthalf", "mnistsump", "cub", "boia"},
        default="mnistadd",
        help="Dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--f1y-threshold",
        type=float,
        default=0.98,
        help="F1Y threshold for early stopping",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Number of max epochs"
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=5,
        help="Percentage of number of supervised examples per step",
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
        default="adam",
        choices=["adam", "sdg"],
        help="Optimizer for training.",
    )


def test_parser(parser):
    # models
    configure_global_arguments(parser)

    subparsers = parser.add_subparsers(dest="model")
    resnet18.configure_subparsers(subparsers)
    lenet.configure_subparsers(subparsers)
    linear.configure_subparsers(subparsers)


def configure_subparsers(subparsers):
    """Configure the subparsers."""
    parser = subparsers.add_parser(
        "active",
        help="Active Learning",
    )
    test_parser(parser)
    parser.set_defaults(func=main)


def active_learning(
    intial_state,
    criterion,
    step_size,
    max_epochs,
    f1y_threshold,
    device,
    data_loader,
    val_dl,
    test_dl,
    n_images,
    output_dim,
    input_dim,
    concept_dim,
    args,
):
    concept_entropy_history, id_concept_entropy_history = [], []
    f1c_history, id_f1c_history = [], []
    f1y_history, id_f1y_history = [], []
    yece_history, id_yece_history = [], []
    cece_history, id_cece_history = [], []
    H_per_value_history = []

    # size of the dataset
    dataset_size = len(data_loader.dataset)

    # elements for the active learning setup
    supervised_indices = set()
    unsupervised_indices = set(range(dataset_size))

    fraction_step_size = step_size / 100.0

    iteration = 1
    model = None

    while unsupervised_indices:
        if iteration > 1:
            num_samples = int(dataset_size * fraction_step_size)
            num_samples = min(num_samples, len(unsupervised_indices))

            if num_samples == 0:
                break

            random_supervision = random.sample(
                list(unsupervised_indices),
                num_samples,
            )

            supervised_indices.update(random_supervision)
            unsupervised_indices.difference_update(random_supervision)

        log(
            f"Active Learning Iteration {iteration}: Supervised examples {len(supervised_indices)} over {dataset_size} total. That is {len(supervised_indices)/dataset_size:.4f} fraction.",
            "INFO",
        )

        model, val_f1, val_c_f1, val_H_c, val_yece, val_cece = active_step(
            initial_state=intial_state,
            criterion=criterion,
            f1y_threshold=f1y_threshold,
            device=device,
            data_loader=data_loader,
            val_dl=val_dl,
            supervised_indices=supervised_indices,
            max_epochs=max_epochs,
            n_images=n_images,
            output_dim=output_dim,
            input_dim=input_dim,
            concept_dim=concept_dim,
            args=args,
        )

        id_yece_history.append(val_yece)
        id_cece_history.append(val_cece)
        id_f1c_history.append(val_c_f1)
        id_f1y_history.append(val_f1)
        id_concept_entropy_history.append(val_H_c)

        model.eval()

        # Evaluate on test set
        test_loss, test_f1, test_c_f1, H_c, H_per_value, yece, cece, _ = (
            compute_statistics(model, test_dl, criterion, device, is_train=False)
        )

        f1c_history.append(test_c_f1)
        f1y_history.append(test_f1)
        yece_history.append(yece)
        cece_history.append(cece)
        concept_entropy_history.append(H_c)
        H_per_value_history.append(H_per_value)

        log(
            f"Current supervised example: {len(supervised_indices)}, Test Loss: {test_loss:.4f} - Test F1: {test_f1:.4f} - Test C F1: {test_c_f1:.4f} - H(C|X): {H_c:.4f} - Y ECE: {yece:.4f} - C ECE: {cece:.4f}"
        )

        iteration += 1
        log(
            f"Active Learning Iteration {iteration}: Supervised examples {len(supervised_indices)}"
        )

        print(len(unsupervised_indices))

    return (
        np.array(concept_entropy_history),
        np.array(H_per_value_history),
        np.array(f1c_history),
        np.array(f1y_history),
        np.array(yece_history),
        np.array(cece_history),
        np.array(id_concept_entropy_history),
        np.array(id_f1c_history),
        np.array(id_f1y_history),
        np.array(id_yece_history),
        np.array(id_cece_history),
        model,
    )


def active_step(
    initial_state,
    criterion,
    f1y_threshold,
    device,
    data_loader,
    val_dl,
    supervised_indices,
    max_epochs,
    n_images,
    output_dim,
    input_dim,
    concept_dim,
    args,
):

    epoch = 0
    val_f1, val_c_f1, H_c, yece, cece = 0.0, 0.0, 0.0, 0.0, 0.0
    supervised_set = set(supervised_indices)

    model, initial_state = load_single_cbm(
        n_images=n_images,
        output_dim=output_dim,
        input_dim=input_dim,
        concept_dim=concept_dim,
        args=args,
        device=device,
        initial_state=initial_state,
    )

    optimizer = OptimizerFactory.create_optimizer(
        args.opt, model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

    while val_f1 < f1y_threshold and epoch < max_epochs:
        model.train()

        global_idx = 0
        # Train model on selected concepts
        for i, (x, concepts, labels) in enumerate(data_loader):
            batch_size = x.size(0)
            batch_indices = list(range(global_idx, global_idx + batch_size))
            global_idx += batch_size

            # Determine which examples in this batch are supervised
            supervised_mask = torch.tensor(
                [idx in supervised_set for idx in batch_indices], dtype=torch.bool
            )

            # move to device
            x = x.to(device)
            concepts = concepts.to(device)
            labels = labels.to(device)

            # prediction
            output, conc_pred = model(x)

            if isinstance(criterion, torch.nn.NLLLoss):
                loss = criterion(output.log(), labels)
            else:
                loss = criterion(output, labels)

            concept_loss = 0.0

            if supervised_mask.sum() > 0:
                for i in range(conc_pred[supervised_mask].size(1)):
                    concept_loss += torch.nn.functional.nll_loss(
                        conc_pred[supervised_mask, i, :].log(),
                        concepts[supervised_mask, i],
                    )
                concept_loss /= conc_pred.size(1)
                concept_loss = args.concept_supervision * concept_loss
                log(f"Concept supervision loss: {concept_loss.item():.4f}", "DEBUG")

            loss += concept_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch += 1

        # Evaluate on validation set
        model.eval()

        # Evaluate on test set
        val_loss, val_f1, val_c_f1, H_c, H_per_value, yece, cece, _ = (
            compute_statistics(model, val_dl, criterion, device, is_train=False)
        )

        log(
            f"Epoch {epoch}, Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f} - Val C F1: {val_c_f1:.4f} - H(C|X): {H_c:.4f} - Y ECE: {yece:.4f} - C ECE: {cece:.4f}"
        )

    return model, val_f1, val_c_f1, H_c, yece, cece


def load_single_cbm(
    n_images, output_dim, input_dim, concept_dim, args, device, initial_state=None
):
    base_model = NetworkFactory.get_network(args.model, input_dim, concept_dim, args)
    model = CBM(
        n_images,
        base_model,
        args.entangled,
        concept_dim,
        output_dim,
        args.dataset,
        True,
    )
    model.to(device)

    if initial_state is not None:
        model.load_state_dict(initial_state)
    else:
        initial_state = copy.deepcopy(model.state_dict())

    return model, initial_state


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
        label_aggregator,
        criterion,
    ) = DatasetFactory.get_dataset(args.dataset, active=True)

    train_dl, val_dl, test_dl = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size, shuffle_val=False
    )

    log("Evaluating the CBM", "INFO")

    _, intial_state = load_single_cbm(
        n_images, output_dim, input_dim, concept_dim, args, device
    )

    (
        concept_entropy_history,
        H_per_value_history,
        f1c_history,
        f1y_history,
        yece_history,
        cece_history,
        id_concept_entropy_history,
        id_f1c_history,
        id_f1y_history,
        id_yece_history,
        id_cece_history,
        model,
    ) = active_learning(
        intial_state=intial_state,
        criterion=criterion,
        step_size=args.step_size,
        max_epochs=args.max_epochs,
        f1y_threshold=args.f1y_threshold,
        device=device,
        data_loader=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        n_images=n_images,
        output_dim=output_dim,
        input_dim=input_dim,
        concept_dim=concept_dim,
        args=args,
    )

    if not args.dry_run:
        path = args.output_dir_path / f"{experiment_name}"

        log(f"> Plots saved {path}", "INFO")

        # OOD plots
        save_line_plot(
            concept_entropy_history,
            output_path=f"{path}_active_concepts.pdf",
            xlabel="Step",
            ylabel="Concept Entropy",
            title="OOD Concept Entropy Over AL Steps",
        )

        save_line_plot(
            f1c_history.reshape(-1, 1),
            output_path=f"{path}_active_f1_c.pdf",
            xlabel="Step",
            ylabel="F1 Score (Concepts)",
            title="OOD Concept F1 Over AL Steps",
        )

        save_line_plot(
            f1y_history.reshape(-1, 1),
            output_path=f"{path}_active_f1_y.pdf",
            xlabel="Step",
            ylabel="F1 Score (Labels)",
            title="OOD Labels F1 Over AL Steps",
        )

        save_line_plot(
            yece_history.reshape(-1, 1),
            output_path=f"{path}_active_ece_y.pdf",
            xlabel="Step",
            ylabel="ECE (Labels)",
            title="OOD ECE Over AL Steps",
        )

        save_line_plot(
            cece_history.reshape(-1, 1),
            output_path=f"{path}_active_ece_c.pdf",
            xlabel="Step",
            ylabel="ECE (Concepts)",
            title="OOD ECE Over AL Steps",
        )

        # In-distribution plots
        save_line_plot(
            id_concept_entropy_history,
            output_path=f"{path}_active_concepts_id.pdf",
            xlabel="Step",
            ylabel="Concept Entropy",
            title="ID Concept Entropy Over AL Steps",
        )

        save_line_plot(
            id_f1c_history.reshape(-1, 1),
            output_path=f"{path}_active_f1_c_id.pdf",
            xlabel="Step",
            ylabel="F1 Score (Concepts)",
            title="ID Concept F1 Over AL Steps",
        )

        save_line_plot(
            id_f1y_history.reshape(-1, 1),
            output_path=f"{path}_active_f1_y_id.pdf",
            xlabel="Step",
            ylabel="F1 Score (Labels)",
            title="ID Labels F1 Over AL Steps",
        )

        save_line_plot(
            id_yece_history.reshape(-1, 1),
            output_path=f"{path}_active_ece_y_id.pdf",
            xlabel="Step",
            ylabel="ECE (Labels)",
            title="ID ECE Over AL Steps",
        )

        save_line_plot(
            id_cece_history.reshape(-1, 1),
            output_path=f"{path}_active_ece_c_id.pdf",
            xlabel="Step",
            ylabel="ECE (Concepts)",
            title="ID ECE Over AL Steps",
        )

        entropy_plot_over_time(
            entropy_values=H_per_value_history,
            concept_names=[f"{i}" for i in range(H_per_value_history.shape[-1])],
            output_path=f"{path}_active_c_entropy_per_value.pdf",
            title="Concept Entropy",
        )

        log("> Collecting predictions...", "INFO")

        (
            all_labels,
            all_preds,
            all_g,
            all_c,
            concept_permutations,
            all_label_prob,
            all_concept_prob,
        ) = collect_predictions(model, test_dl, device)
        log("> Predictions collected.", "INFO")

        log("> Label confusion matrix...", "INFO")
        plot_confusion_matrix(
            all_labels,
            all_preds,
            class_names,
            "Label confusion matrix",
            str(args.output_dir_path / f"{experiment_name}.y_cm_active.pdf"),
        )

        log("> Concept confusion matrix...", "INFO")
        plot_confusion_matrix(
            all_g,
            all_c,
            concept_names,
            "Concept confusion matrix",
            str(args.output_dir_path / f"{experiment_name}.c_cm_active.pdf"),
        )


if __name__ == "__main__":
    main()
