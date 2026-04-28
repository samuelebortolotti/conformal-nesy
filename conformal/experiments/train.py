"""Trains a NN and saves statistics and output"""

import torch
import csv

import numpy as np

from sklearn.metrics import f1_score
from conformal.general_utils import log
from conformal.utils.factories import (
    DatasetFactory,
    OptimizerFactory,
    NetworkFactory,
    NeSyFactory,
)
from conformal.datasets.loaders import create_dataloaders
from conformal.statistics.statistics import Statistics, Results
from conformal.utils.visualization import plot_confusion_matrix
from conformal.experiments.utils import collect_predictions
from conformal.statistics.metrics import compute_statistics
from conformal.datasets import (
    boia,
    chx,
    derma,
    mnistadd,
    mnisthalf,
    mnistsump,
    mnistaddn,
    mnistevenodd,
    rival,
    cifar,
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


def train_parser(parser):
    # models
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
        "train",
        help="Trains a nn model on a dataset",
    )
    train_parser(parser)
    parser.set_defaults(func=main)


def train_epoch(
    model, train_dl, optimizer, criterion, device, args, concept_weights, label_weights,
    concept_sup_weight=None,
):
    """
    Train model for one epoch.
    """
    if concept_sup_weight is None:
        concept_sup_weight = args.concept_supervision

    running_loss = 0.0
    all_preds = []
    all_labels = []

    all_g = []
    all_c = []

    model.train()  # Ensure the model is in training mode
    for data, concepts, target in train_dl:
        data = data.to(device)
        target = target.to(device)
        concepts = concepts.to(device)

        optimizer.zero_grad()
        output, conc_pred, extra = model(data, eval=False)

        # NeSy loss specific
        loss = model.compute_loss(
            args.dataset,
            criterion,
            conc_pred,
            concepts,
            output,
            target,
            label_weights,
            extra,
        )

        # Add concept supervision loss if specified
        if concept_sup_weight > 0:
            concept_loss = 0.0

            if args.dataset in ["chx", "boia", "derma", "rival", "cifar"]:
                for i in range(conc_pred.size(1)):
                    concept_loss += torch.nn.functional.nll_loss(
                        conc_pred[:, 0, i, :].log(),
                        concepts[:, i].long(),
                        weight=concept_weights[i] if concept_weights else None,
                    )

            else:
                for i in range(conc_pred.size(1)):
                    concept_loss += torch.nn.functional.nll_loss(
                        conc_pred[:, i, :].log(),
                        concepts[:, i].long(),
                        weight=(
                            concept_weights[i] if concept_weights is not None else None
                        ),
                    )
            concept_loss /= concepts.size(1)
            concept_loss = concept_sup_weight * concept_loss
            log(f"Concept supervision loss: {concept_loss.item():.4f}", "DEBUG")
            loss += concept_loss

        if loss <= 0:
            log(f"Loss should be greater than zero: {loss}", "CRITICAL")
            exit(1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        all_preds.append(output.argmax(dim=-1).cpu().numpy())
        all_labels.append(target.cpu().numpy())

        all_c.append(conc_pred.argmax(dim=-1).cpu().numpy())
        all_g.append(concepts.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_g = np.concatenate(all_g)
    all_c = np.concatenate(all_c)

    if args.dataset == "boia":
        train_f1 = 0.0

        for idx in range(all_labels.shape[1]):
            present_l = np.unique(all_labels[:, idx])
            train_f1 += f1_score(
                all_labels[:, idx],
                all_preds[:, idx],
                labels=present_l,
                average="macro",
            )
        train_f1 /= all_labels.shape[1]

    else:
        present_l = np.unique(all_labels)
        train_f1 = f1_score(all_labels, all_preds, labels=present_l, average="macro")

    if args.dataset in ["boia", "chx", "derma", "rival", "cifar"]:
        train_c_f1 = 0.0
        for idx in range(all_g.shape[1]):
            present_c = np.unique(all_g[:, idx])
            train_c_f1 += f1_score(
                all_g[:, idx],
                all_c[:, 0, idx],
                labels=present_c,
                average="macro",
            )
        train_c_f1 /= all_g.shape[1]
    else:
        present_c = np.unique(all_g)
        train_c_f1 = f1_score(
            all_g.flatten(),
            all_c.flatten(),
            labels=present_c,
            average="macro",
        )
    return running_loss / len(train_dl), train_f1, train_c_f1


def train(
    model,
    train_dl,
    val_dl,
    epochs,
    optimizer,
    device,
    criterion,
    args,
    experiment_name,
    concept_weights,
    label_weights,
):
    statistics = Statistics()
    model = model.to(device)

    warmup_end = max(1, epochs // 4)

    for epoch in range(epochs):
        # Concept supervision warm-up: only for LTN with full concept supervision
        if args.nesy == "ltn" and args.concept_supervision == 1.0:
            if epoch < warmup_end:
                concept_sup_weight = 0.0
            else:
                concept_sup_weight = (epoch - warmup_end) / max(1, epochs - warmup_end)
        else:
            concept_sup_weight = args.concept_supervision

        # p scheduler: linearly ramp from 1 to args.p over all epochs (LTN only)
        if args.nesy == "ltn" and args.p > 1:
            scheduled_p = round(1 + (args.p - 1) * epoch / max(1, epochs - 1))
            model.set_p(scheduled_p)

        train_loss, train_f1, train_c_f1 = train_epoch(
            model,
            train_dl,
            optimizer,
            criterion,
            device,
            args,
            concept_weights,
            label_weights,
            concept_sup_weight=concept_sup_weight,
        )

        val_loss, val_f1, val_c_f1, H_c, H_c_per_value, yece, cece, _, _, _, _, _ = (
            compute_statistics(
                model,
                args.dataset,
                val_dl,
                criterion,
                device,
                is_train=False,
                multiclass=(
                    False
                    if args.dataset not in ["boia", "chx", "derma", "cifar", "rival"]
                    else True
                ),
                multilabel=False if args.dataset not in ["boia"] else True,
                is_dpl=args.nesy == "dpl",
            )
        )
        statistics.log(
            epoch, train_loss, train_f1, val_f1, train_c_f1, val_c_f1, val_loss, model
        )

        log(
            f"Epoch {epoch+1:3}/{epochs:3} - Train Loss: {train_loss:2.4f} - Train F1: {train_f1:2.4f} - Val Loss: {val_loss:2.4f} - Val F1: {val_f1:2.4f} - Train C F1: {train_c_f1:2.4f} - Val C F1: {val_c_f1:2.4f} - Val H(C|X): {H_c:2.4f} - Val YECE: {yece:2.4f} - Val CECE: {cece:2.4f}",
            "INFO",
        )

    return statistics, statistics.best_model


def evaluate_and_log_model(
    model,
    experiment_name,
    test_dl,
    device,
    criterion,
    class_names,
    concept_names,
    concept_dim,
    n_images,
    logic,
    args,
    results_output_h,
    stats_output_h,
    statistics,
):
    if not args.dry_run:
        log("> Model saved...", "INFO")
        model_path = args.output_dir_path / f"{experiment_name}.{args.model_path}.pth"
        torch.save(statistics.best_model, str(model_path))

    log(f"> Load the best model: f1 = {statistics.best_f1}", "INFO")
    model.load_state_dict(statistics.best_model)

    test_loss, test_f1, test_c_f1, H_c, H_c_per_value, yece, cece, _, _, _, _, _ = (
        compute_statistics(
            model,
            args.dataset,
            test_dl,
            criterion,
            device,
            is_train=False,
            multiclass=(
                False
                if args.dataset not in ["boia", "chx", "derma", "rival", "cifar"]
                else True
            ),
            multilabel=False if args.dataset not in ["boia"] else True,
            is_dpl=args.nesy == "dpl",
        )
    )
    log(
        f"Test Loss: {test_loss:.4f} - Test F1: {test_f1:.4f} - Test C F1: {test_c_f1:.4f}  - H(C|X): {H_c:.4f} - YECE(C|X): {yece:.4f} - CECE(C|X): {cece:.4f}"
    )

    if not args.dry_run:
        log("> Collecting predictions...", "INFO")
        all_labels, all_preds, all_g, all_c, _, all_label_prob, all_concept_prob = (
            collect_predictions(
                model,
                args.dataset,
                test_dl,
                device,
                args.nesy,
                multiclass=(
                    False
                    if args.dataset not in ["boia", "chx", "derma", "rival", "cifar"]
                    else True
                ),
                multilabel=False if args.dataset not in ["boia"] else True,
                is_dpl=args.nesy == "dpl",
            )
        )
        log("> Predictions collected.", "INFO")

        log("> Label confusion matrix...", "INFO")

        plot_confusion_matrix(
            all_labels,
            all_preds,
            class_names,
            "Label confusion matrix",
            str(args.output_dir_path / f"{experiment_name}.label_confusion_matrix.pdf"),
            multilabel=args.dataset == "boia",
        )

        log("> Concept confusion matrix...", "INFO")
        plot_confusion_matrix(
            all_g,
            all_c,
            concept_names,
            "Concept confusion matrix",
            str(
                args.output_dir_path / f"{experiment_name}.concept_confusion_matrix.pdf"
            ),
            multilabel=(
                True
                if args.dataset in ["boia", "chx", "derma", "rival", "cifar"]
                else False
            ),
        )

        res = Results(
            test_loss,
            test_f1,
            yece,
            cece,
            H_c,
            H_c_per_value,
            all_labels.tolist(),
            all_preds.tolist(),
            all_g.tolist(),
            all_c.tolist(),
            all_label_prob.tolist(),
            all_concept_prob.tolist(),
        )

        log("> Writing results...", "INFO")
        writer = csv.writer(results_output_h)
        writer.writerows(res.get_statistics())

        writer = csv.writer(stats_output_h)
        writer.writerows(statistics.get_statistics())

    return test_f1


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

    train_dl, val_dl, test_dl = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size
    )

    log(f"Dataset {args.dataset} loaded.", "INFO")
    log(f"Number of training samples: {len(train_ds)}", "INFO")
    log(f"Number of validation samples: {len(val_ds)}", "INFO")
    log(f"Number of test samples: {len(test_ds)}", "INFO")

    log("Training model", "INFO")

    model = NetworkFactory.get_network(
        args.model, input_dim, concept_dim, args, n_images=n_images
    )
    model = NeSyFactory.get_nesy_model(
        args.nesy, n_images, model, concept_dim, output_dim, device, logic, args
    )

    optimizer = OptimizerFactory.create_optimizer(
        args.opt, model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

    log("> Start training...", "INFO")

    statistics, statistics.best_model = train(
        model,
        train_dl,
        val_dl,
        args.epochs,
        optimizer,
        device,
        criterion,
        args,
        experiment_name,
        concept_weights,
        label_weights,
    )

    log("> Training completed.", "INFO")

    log("> Evaluating and logging model...", "INFO")

    test_f1 = evaluate_and_log_model(
        model,
        experiment_name,
        test_dl,
        device,
        criterion,
        class_names,
        concept_names,
        concept_dim,
        n_images,
        logic,
        args,
        results_output_h,
        stats_output_h,
        statistics,
    )

    log("> Evaluation and logging completed.", "INFO")

    log("> Returning best F1 score on labels in the validation set...", "INFO")

    return statistics.best_f1
