import math
import torch

from torch import nn
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from conformal.utils.alignment import align_knowledge_input
from conformal.general_utils import log
from conformal.utils.other import outer_product


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
    permutation=None,
    is_dpl=False,
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
    circuit = None

    for data, concepts, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output, conc_pred, extra = model(data)

        if permutation is not None:
            conc_pred = align_knowledge_input(
                (
                    conc_pred.detach().cpu().numpy()
                    if dataset.startswith("mnist")
                    else conc_pred.squeeze(1).detach().cpu().numpy()
                ),
                permutation,
            )
            conc_pred = torch.tensor(conc_pred, device=device)
            conc_pred = (
                conc_pred if dataset.startswith("mnist") else conc_pred.unsqueeze(1)
            )

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

        if is_dpl and circuit is None:
            circuit = extra.clone().detach().cpu()

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

    if is_dpl:
        # Consistent y and c given by argmax (c, y) p(c, y | x)
        worlds = (
            outer_product(torch.tensor(all_conc_pred).squeeze(1))
            if dataset in ["chx", "derma", "rival", "cifar"]
            else outer_product(torch.tensor(all_conc_pred))
        )  # p(c | x)

        # p(y, c | x) = p(y | c) * p(c | x)
        joint = worlds.unsqueeze(2) * circuit.unsqueeze(0)
        B, C, Y = (
            joint.shape
        )  # B: batch size, C: number of concepts, Y: number of labels
        joint_flat = joint.view(B, -1)  # shape (B, C*Y)
        idx = joint_flat.argmax(dim=1)  # (876,)

        c_idx = idx // Y  # integer division to get concept index
        y_idx = idx % Y  # modulus to get label index

        # to numpy
        c_idx = c_idx.cpu().numpy()
        y_idx = y_idx.cpu().numpy()

        # override all_preds and all_c with the consistent predictions
        all_preds = y_idx
        all_c = c_idx

        num_concepts = (
            all_conc_pred.shape[2]
            if dataset in ["chx", "derma", "rival", "cifar"]
            else all_conc_pred.shape[1]
        )
        all_c = (
            c_idx[:, None] >> np.arange(num_concepts - 1, -1, -1)
        ) & 1  # convert back to binary concept vector

    else:
        # find predicted concept and probability distribution: argmax p(c|x) e.g. ltn
        if all_conc_pred.ndim == 2:
            all_c = all_conc_pred.argmax(axis=1)
        elif all_conc_pred.ndim == 3:
            all_c = all_conc_pred.argmax(axis=2)
        elif all_conc_pred.ndim == 4:
            all_c = all_conc_pred.argmax(axis=3).squeeze(1)

    if multilabel:
        f1, acc, rec = 0.0, 0.0, 0.0
        for idx in range(all_labels.shape[1]):
            f1 += f1_score(all_labels[:, idx], all_preds[:, idx], average="macro")
            acc += accuracy_score(all_labels[:, idx], all_preds[:, idx], normalize=True)
            rec += recall_score(all_labels[:, idx], all_preds[:, idx], average="macro")
        f1 /= all_labels.shape[1]
        acc /= all_labels.shape[1]
        rec /= all_labels.shape[1]
    else:
        f1 = f1_score(all_labels, all_preds, labels=present_labels, average="macro")
        acc = accuracy_score(all_labels, all_preds, normalize=True)
        rec = recall_score(
            all_labels, all_preds, labels=present_labels, average="macro"
        )

    if multiclass:
        c_f1, c_acc, c_rec = 0.0, 0.0, 0.0
        for idx in range(all_g.shape[1]):
            col_present_concepts = np.unique(all_g[:, idx])
            c_f1 += f1_score(
                all_g[:, idx],
                all_c[:, idx],
                labels=col_present_concepts,
                average="macro",
            )
            c_acc += accuracy_score(
                all_g[:, idx],
                all_c[:, idx],
                normalize=True,
            )
            c_rec += recall_score(
                all_g[:, idx],
                all_c[:, idx],
                labels=col_present_concepts,
                average="macro",
            )
        c_f1 /= all_g.shape[1]
        c_acc /= all_g.shape[1]
        c_rec /= all_g.shape[1]
    else:
        c_f1 = f1_score(
            all_g.flatten(),
            all_c.flatten(),
            labels=present_concepts,
            average="macro",
        )
        c_acc = accuracy_score(
            all_g.flatten(),
            all_c.flatten(),
            normalize=True,
        )
        c_rec = recall_score(
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
        acc,
        rec,
        c_acc,
        c_rec,
    )


def conformal_metrics(prediction_tuples, true_labels, ignore_token=-1):
    """
    Compute conformal metrics for concept combinations (tuples).

    Args:
        prediction_tuples: List of np.ndarrays [Samples][Combinations, Concepts]
        true_labels: np.ndarray [Samples, Concepts]
    """
    N = len(true_labels)

    coverage_total = 0.0
    set_size_total = 0.0

    for i in range(N):
        sample_tuples = prediction_tuples[i]
        ground_truth = true_labels[i]

        # print(ground_truth)
        # print(len(sample_tuples), sample_tuples, len(sample_tuples[0]))

        assert (
            len(ground_truth.shape) == 1
        ), f"Ground-truth: dim of the worlds. Got {ground_truth.shape, ground_truth.tolist()}"
        assert (
            len(sample_tuples.shape) == 2 or len(sample_tuples) == 0
        ), f"Predictions: must be number of elements in the conformal set, size of the world. Got {sample_tuples.shape, sample_tuples.tolist(), len(sample_tuples)}"

        # Filter tuples: ignore any tuple that is all ignore tokens
        valid_mask = ~np.all(sample_tuples == ignore_token, axis=-1)
        valid_tuples = sample_tuples[valid_mask]

        # Check if ground truth exists in valid tuples
        if valid_tuples.size > 0:
            if np.any(np.all(ground_truth == valid_tuples, axis=-1)):
                coverage_total += 1

        # print(sample_tuples, np.all(sample_tuples == ignore_token, axis=-1), len(valid_tuples))

        set_size_total += len(valid_tuples)

    # Calculate final averages
    coverage = coverage_total / N
    avg_set_size = set_size_total / N

    return coverage, avg_set_size


def prediction_consistency(concept_tuples, label_sets, logic, EMPTY_TOKEN=-1):
    """
    Consistency metrics for the predicted concept tuples and the predicted label sets.
    """
    N = len(concept_tuples)
    total_concepts = 0
    consistent_concepts = 0
    total_labels = 0
    covered_labels = 0

    for i in range(N):
        tuples_i = concept_tuples[i]
        labels_i = label_sets[i]

        # Some concepts are empty
        concept_empty_mask = np.any(tuples_i == EMPTY_TOKEN, axis=-1)

        assert (
            len(tuples_i.shape) == 2
        ), f"Concepts: should be size conformal, dim of the worlds. Got {tuples_i.shape, tuples_i.tolist()}"
        assert (
            len(labels_i.shape) == 2 or len(labels_i) == 0
        ), f"Labels: should be size conformal, size of the world. Got {labels_i.shape, labels_i.tolist(), len(labels_i)}"

        # filter the valid concepts
        valid_tuples = tuples_i[~concept_empty_mask]

        # Concept consistency: fraction of tuples producing at least one label in predicted label set
        for t in valid_tuples:
            if logic.multi_set_logic is None:
                derived_label = logic.forward(t.reshape(1, -1))
                derived_label = np.array(derived_label).ravel()
            else:
                derived_label = logic.forward_multi_set(t.reshape(1, -1))
                derived_label = np.unique(np.concatenate(derived_label))
            total_concepts += 1
            if np.any(np.isin(derived_label, labels_i)):
                consistent_concepts += 1
            else:
                log(
                    f"Inconsistent concept tuple: {t} -> derived label {derived_label} not in predicted labels {labels_i} for sample {i}",
                    "DEBUG",
                )
                log(f"Tuples: {valid_tuples.tolist()}", "DEBUG")
                log(f"Labels: {labels_i.tolist()}", "DEBUG")

        # Label consistency: fraction of predicted labels covered by at least one concept tuple
        total_labels += len(labels_i)
        for l in labels_i:
            covered = False
            for t in valid_tuples:
                derived_label = logic.forward(t.reshape(1, -1))
                derived_label = np.array(derived_label).ravel()
                if np.any(np.isin(l, derived_label)):
                    covered = True
                    break
            if covered:
                covered_labels += 1
            else:
                log(
                    f"Inconsistent label: {l} not covered by any concept tuple for sample {i}",
                    "DEBUG",
                )
                log(f"Tuples: {valid_tuples.tolist()}", "DEBUG")
                log(f"Labels: {labels_i.tolist()}", "DEBUG")

    log(
        f"Total Concepts: {total_concepts}, Consistent Concepts: {consistent_concepts}",
        "INFO",
    )
    log(f"Total Labels: {total_labels}, Covered Labels: {covered_labels}", "INFO")

    concept_consistency = (
        consistent_concepts / total_concepts if total_concepts > 0 else 1.0
    )
    label_consistency = covered_labels / total_labels if total_labels > 0 else 1.0

    return concept_consistency, label_consistency


def conditional_conformal_metrics(
    concept_tuples, label_sets, true_concepts, true_labels
):
    """
    delta_ab = P(c* in Gamma | y* in Upsilon)
    delta_de = P(y* in Upsilon | c* in Gamma)
    """
    N = len(concept_tuples)

    ab_numerator, ab_denominator = 0, 0
    de_numerator, de_denominator = 0, 0

    for i in range(N):
        tuples_i = concept_tuples[i]  # (K, n_concepts)
        labels_i = label_sets[i]  # (M, 1) or (M,)
        c_star = true_concepts[i]  # (n_concepts,)
        y_star = true_labels[i]

        flat_labels = labels_i.ravel()

        y_covered = np.isin(y_star, flat_labels)
        c_covered = np.any(np.all(tuples_i == c_star, axis=1))

        if y_covered:
            ab_denominator += 1
            if c_covered:
                ab_numerator += 1

        if c_covered:
            de_denominator += 1
            if y_covered:
                de_numerator += 1

    delta_ab = ab_numerator / ab_denominator if ab_denominator > 0 else float("nan")
    delta_de = de_numerator / de_denominator if de_denominator > 0 else float("nan")

    return delta_ab, delta_de
