import math
import torch

from torch import nn
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix


def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    accuracies = predictions == labels

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return ece


def compute_statistics(
    model,
    dataset,
    data_loader,
    criterion,
    device,
    is_train=True,
    multiclass=False,
    multilabel=False,
):
    """
    Compute loss and F1 score for the dataset (train or validation).
    """
    model.eval() if not is_train else model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_g = []
    all_conc_pred = []
    all_probs = []

    for data, concepts, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output, conc_pred, extra = model(data)

        loss = model.compute_loss(
            dataset=dataset,
            criterion=criterion,
            conc_pred=conc_pred,
            concepts=concepts,
            output=output,
            target=target,
            label_weights=None,
            extra=extra,
        )

        if isinstance(criterion, nn.NLLLoss):
            probs = output.detach().cpu().numpy()
        else:
            probs = torch.softmax(output, dim=-1).detach().cpu().numpy()

        running_loss += loss.item()

        all_preds.append(output.argmax(dim=-1).cpu().numpy())
        all_labels.append(target.cpu().numpy())
        all_probs.append(probs)

        all_conc_pred.append(conc_pred.detach().cpu().numpy())
        all_g.append(concepts.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_conc_pred = np.concatenate(all_conc_pred)
    all_g = np.concatenate(all_g)

    # Identify present labels and present concepts
    present_labels = np.unique(all_labels)
    present_concepts = np.unique(all_g)

    # find predicted concept and probability distribution
    if all_conc_pred.ndim == 2:
        all_c = all_conc_pred.argmax(axis=1)
    elif all_conc_pred.ndim == 3:
        all_c = all_conc_pred.argmax(axis=2)
    elif all_conc_pred.ndim == 4:
        all_c = all_conc_pred.argmax(axis=3).squeeze(1)

    if multilabel:
        f1 = 0.0
        for idx in range(all_labels.shape[1]):
            f1 += f1_score(all_labels[:, idx], all_preds[:, idx], average="macro")
        f1 /= all_labels.shape[1]
    else:
        f1 = f1_score(all_labels, all_preds, labels=present_labels, average="macro")

    if multiclass:
        c_f1 = 0.0
        for idx in range(all_g.shape[1]):
            col_present_concepts = np.unique(all_g[:, idx])
            c_f1 += f1_score(
                all_g[:, idx],
                all_c[:, idx],
                labels=col_present_concepts,
                average="macro",
            )
        c_f1 /= all_g.shape[1]
    else:
        c_f1 = f1_score(
            all_g.flatten(),
            all_c.flatten(),
            labels=present_concepts,
            average="macro",
        )

    if not multiclass:
        cm = confusion_matrix(
            all_g.flatten(),
            all_c.flatten(),
            labels=np.arange(all_conc_pred.shape[2]),
        )
    else:
        cm = all_c.astype(float).T @ all_g.astype(float)

    # Print as a 10x10 matrix
    # print("Confusion Matrix:")
    # print(cm)

    if multilabel:
        y_ece = 0.0
        for idx in range(all_labels.shape[1]):
            y_ece += compute_ece(all_probs[:, idx], all_labels[:, idx], n_bins=15)
        y_ece /= all_labels.shape[1]
    else:
        y_ece = compute_ece(all_probs, all_labels, n_bins=15)

    if multiclass:
        c_ece = 0.0
        for idx in range(all_conc_pred.shape[1]):
            c_ece += compute_ece(
                all_conc_pred[:, idx].argmax(axis=-1), all_g[:, idx], n_bins=15
            )
        c_ece /= all_conc_pred.shape[1]
    else:
        if all_conc_pred.ndim == 3:
            N, M, C = all_conc_pred.shape
            all_conc_pred = all_conc_pred.reshape(-1, C)
            all_g = all_g.reshape(-1)
        c_ece = compute_ece(all_conc_pred, all_g, n_bins=15)

    if multiclass:
        H_c_matrix = -all_conc_pred * np.log(all_conc_pred + 1e-12) - (
            1 - all_conc_pred
        ) * np.log(1 - all_conc_pred + 1e-12)
        H_c = H_c_matrix.mean()
        H_per_value = H_c_matrix.mean(axis=0)
    else:
        H_c = -np.sum(all_conc_pred * np.log(all_conc_pred + 1e-12), axis=1) / math.log(
            all_conc_pred.shape[1]
        )
        H_c = H_c.mean()

        # Per-concept-value entropy
        N, C = all_conc_pred.shape
        H_per_value = -np.mean(
            all_conc_pred * np.log(all_conc_pred + 1e-12), axis=0
        ) / np.log(
            C
        )  # shape (C,)

    return (
        running_loss / len(data_loader),
        f1,
        c_f1,
        H_c,
        H_per_value,
        y_ece,
        c_ece,
        None,
    )


def conformal_metrics(prediction_tuples, true_labels):
    """
    Compute conformal metrics for concept combinations (tuples).

    Args:
        prediction_tuples: List of np.ndarrays [Samples][Combinations, Concepts]
        true_labels: np.ndarray [Samples, Concepts]
    """
    true_labels = np.array(true_labels)
    N = len(true_labels)

    coverage_total = 0.0
    set_size_total = 0.0

    # Loop through the examples (samples)
    for i in range(N):
        # This is your matrix of all valid conformal combinations for this sample
        sample_tuples = prediction_tuples[i]

        # The ground truth combination we are looking for
        ground_truth = true_labels[i]

        # Check the presence of the true combination in the predicted tuples
        if sample_tuples.size > 0:
            if np.any(np.all(sample_tuples == ground_truth, axis=-1)):
                coverage_total += 1

        # Increment total set size by the number of unique tuples predicted
        set_size_total += len(sample_tuples)

    # Calculate final averages
    coverage = coverage_total / N
    avg_set_size = set_size_total / N

    return coverage, avg_set_size
