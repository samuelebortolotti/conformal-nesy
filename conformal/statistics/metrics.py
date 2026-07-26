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
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(is_train)
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

        if is_dpl and circuit is None and not isinstance(extra, tuple):
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

    if is_dpl and circuit is not None:
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

    torch.set_grad_enabled(prev_grad)
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


_BOIA_FS_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8]
_BOIA_L_IDX  = [9, 10, 11, 18, 19, 20]
_BOIA_R_IDX  = [12, 13, 14, 15, 16, 17]


def boia_conformal_metrics(prediction_tuples, true_labels):
    """
    Compute per-group conformal metrics for BOIA.

    Args:
        prediction_tuples: list of (fs_worlds, l_worlds, r_worlds) tuples per sample,
                           where each group array has shape (n_worlds, group_size).
        true_labels: (N, 21) array of ground-truth concept labels.

    Returns:
        dict with keys: coverage_fs, coverage_l, coverage_r, set_size_fs, set_size_l,
                        set_size_r, coverage_avg, set_size_avg
    """
    N = len(prediction_tuples)
    group_idx_list = [_BOIA_FS_IDX, _BOIA_L_IDX, _BOIA_R_IDX]

    coverage_counts = [0, 0, 0]
    set_size_totals = [0.0, 0.0, 0.0]

    for i in range(N):
        fs_worlds, l_worlds, r_worlds = prediction_tuples[i]
        group_worlds = [fs_worlds, l_worlds, r_worlds]
        gt = true_labels[i]
        if hasattr(gt, "ndim") and gt.ndim > 1:
            gt = gt.reshape(-1)

        for g, (grp_idx, grp_worlds) in enumerate(zip(group_idx_list, group_worlds)):
            gt_slice = gt[grp_idx]
            set_size_totals[g] += len(grp_worlds)
            if len(grp_worlds) > 0 and np.any(np.all(gt_slice == grp_worlds, axis=1)):
                coverage_counts[g] += 1

    coverage_fs = coverage_counts[0] / N
    coverage_l  = coverage_counts[1] / N
    coverage_r  = coverage_counts[2] / N
    set_size_fs = set_size_totals[0] / N
    set_size_l  = set_size_totals[1] / N
    set_size_r  = set_size_totals[2] / N

    return {
        "coverage_fs":  coverage_fs,
        "coverage_l":   coverage_l,
        "coverage_r":   coverage_r,
        "set_size_fs":  set_size_fs,
        "set_size_l":   set_size_l,
        "set_size_r":   set_size_r,
        "coverage_avg": (coverage_fs + coverage_l + coverage_r) / 3,
        "set_size_avg": (set_size_fs + set_size_l + set_size_r) / 3,
    }


def _prediction_consistency_boia_groups(concept_tuples, label_sets, logic):
    """
    Compute concept/label consistency for BOIA group-tuple representation.

    concept_tuples: list of (fs_worlds, l_worlds, r_worlds) per sample.
    label_sets: list of (M, 4) label arrays per sample.
    logic: logic object with forward(x) -> (N,4) method.
    """
    group_idx_list = [_BOIA_FS_IDX, _BOIA_L_IDX, _BOIA_R_IDX]
    extract_fns = [
        lambda lv: (int(lv[1] > 0), int(lv[0] > 0)),
        lambda lv: int(lv[2] > 0),
        lambda lv: int(lv[3] > 0),
    ]
    label_fns = [
        lambda l: (int(l[0]), int(l[1])),
        lambda l: int(l[2]),
        lambda l: int(l[3]),
    ]

    total_concepts = 0
    consistent_concepts = 0
    total_labels = 0
    covered_labels = 0

    N = len(concept_tuples)
    for i in range(N):
        group_tuple = concept_tuples[i]
        labels_i = label_sets[i]
        labels_arr = np.asarray(labels_i)

        # Compute achievable bits per group
        achievable = [set(), set(), set()]
        for g, (grp_idx, grp_worlds, extract_fn) in enumerate(
            zip(group_idx_list, group_tuple, extract_fns)
        ):
            for row in grp_worlds:
                eval_vec = np.zeros(21, dtype=int)
                for local_i, gi in enumerate(grp_idx):
                    eval_vec[gi] = row[local_i]
                lv = np.asarray(logic.forward(eval_vec.reshape(1, -1)))
                lv = lv[0] if lv.ndim == 2 else lv.ravel()
                achievable[g].add(extract_fn(lv))

        # Concept consistency: each group row produces a bit matching at least one label
        for g, (grp_idx, grp_worlds, extract_fn, label_fn) in enumerate(
            zip(group_idx_list, group_tuple, extract_fns, label_fns)
        ):
            for row in grp_worlds:
                eval_vec = np.zeros(21, dtype=int)
                for local_i, gi in enumerate(grp_idx):
                    eval_vec[gi] = row[local_i]
                lv = np.asarray(logic.forward(eval_vec.reshape(1, -1)))
                lv = lv[0] if lv.ndim == 2 else lv.ravel()
                bit = extract_fn(lv)
                total_concepts += 1
                if len(labels_arr) == 0 or any(label_fn(l) == bit for l in labels_arr):
                    consistent_concepts += 1

        # Label consistency: each predicted label has all group bits achievable
        total_labels += len(labels_arr)
        for l in labels_arr:
            covered = all(
                label_fn(l) in ach
                for label_fn, ach in zip(label_fns, achievable)
            )
            if covered:
                covered_labels += 1

    log(
        f"Total Concepts: {total_concepts}, Consistent Concepts: {consistent_concepts}",
        "INFO",
    )
    log(f"Total Labels: {total_labels}, Covered Labels: {covered_labels}", "INFO")

    concept_consistency = consistent_concepts / total_concepts if total_concepts > 0 else 1.0
    label_consistency = covered_labels / total_labels if total_labels > 0 else 1.0
    return concept_consistency, label_consistency


def boia_per_group_concept_consistency(concept_tuples, label_sets, logic):
    """Per-group concept consistency for BOIA group-tuple representation.

    Returns {"fs": float, "l": float, "r": float} where each value is the
    fraction of concept worlds in that group whose logic output is consistent
    with at least one predicted label combo.
    """
    group_idx_list = [_BOIA_FS_IDX, _BOIA_L_IDX, _BOIA_R_IDX]
    extract_fns = [
        lambda lv: (int(lv[1] > 0), int(lv[0] > 0)),
        lambda lv: int(lv[2] > 0),
        lambda lv: int(lv[3] > 0),
    ]
    label_fns = [
        lambda l: (int(l[0]), int(l[1])),
        lambda l: int(l[2]),
        lambda l: int(l[3]),
    ]

    totals = [0, 0, 0]
    consistent = [0, 0, 0]

    for i in range(len(concept_tuples)):
        group_tuple = concept_tuples[i]
        labels_arr = np.asarray(label_sets[i])
        for g, (grp_idx, grp_worlds, extract_fn, label_fn) in enumerate(
            zip(group_idx_list, group_tuple, extract_fns, label_fns)
        ):
            for row in grp_worlds:
                eval_vec = np.zeros(21, dtype=int)
                for local_i, gi in enumerate(grp_idx):
                    eval_vec[gi] = row[local_i]
                lv = np.asarray(logic.forward(eval_vec.reshape(1, -1)))
                lv = lv[0] if lv.ndim == 2 else lv.ravel()
                bit = extract_fn(lv)
                totals[g] += 1
                if len(labels_arr) == 0 or any(label_fn(l) == bit for l in labels_arr):
                    consistent[g] += 1

    names = ["fs", "l", "r"]
    return {
        name: consistent[g] / totals[g] if totals[g] > 0 else 1.0
        for g, name in enumerate(names)
    }


_BOIA_LABEL_FS_IDX = [0, 1]
_BOIA_LABEL_L_IDX  = [2]
_BOIA_LABEL_R_IDX  = [3]


def boia_label_group_metrics(label_sets, true_labels):
    """
    Per-group label coverage and set-size for BOIA.

    FS group: label indices [0,1] (STOP, FORWARD), treated as a joint task (k=2 Bonferroni)
    L  group: label index   [2]   (LEFT),  independent task
    R  group: label index   [3]   (RIGHT), independent task

    Args:
        label_sets:   list of (M, 4) arrays — predicted label combos per sample
        true_labels:  (N, 4) ground-truth binary label array

    Returns:
        dict with keys coverage_label_{fs,l,r} and set_size_label_{fs,l,r}
    """
    N = len(label_sets)
    group_idx = [_BOIA_LABEL_FS_IDX, _BOIA_LABEL_L_IDX, _BOIA_LABEL_R_IDX]
    names = ["fs", "l", "r"]

    coverage = [0, 0, 0]
    set_size = [0.0, 0.0, 0.0]

    for i in range(N):
        ls = np.asarray(label_sets[i])          # (M, 4)
        gt = np.asarray(true_labels[i]).ravel()  # (4,)

        for g, idx in enumerate(group_idx):
            gt_slice = gt[idx]
            if ls.ndim == 2 and ls.shape[0] > 0:
                ls_slice = ls[:, idx]
                unique_combos = np.unique(ls_slice, axis=0)
            else:
                unique_combos = np.zeros((0, len(idx)), dtype=int)

            set_size[g] += len(unique_combos)
            if len(unique_combos) > 0 and np.any(np.all(gt_slice == unique_combos, axis=1)):
                coverage[g] += 1

    result = {}
    for g, name in enumerate(names):
        result[f"coverage_label_{name}"] = coverage[g] / N
        result[f"set_size_label_{name}"] = set_size[g] / N
    return result


def boia_per_group_label_consistency(concept_tuples, label_sets, logic):
    """
    Per-group label consistency for BOIA (label → concept direction).

    For each group (FS/L/R): fraction of unique predicted label combos (in that group's
    label dimensions) that are achievable by at least one concept world in that group.

    Args:
        concept_tuples: list of 3-tuples (fs_worlds, l_worlds, r_worlds) per sample
        label_sets:     list of (M, 4) predicted label arrays per sample
        logic:          logic object with forward(vec) -> label activations

    Returns:
        dict with keys label_consistency_{fs,l,r}
    """
    group_idx_list = [_BOIA_FS_IDX, _BOIA_L_IDX, _BOIA_R_IDX]
    label_slice_idx = [_BOIA_LABEL_FS_IDX, _BOIA_LABEL_L_IDX, _BOIA_LABEL_R_IDX]
    extract_fns = [
        lambda lv: (int(lv[1] > 0), int(lv[0] > 0)),
        lambda lv: int(lv[2] > 0),
        lambda lv: int(lv[3] > 0),
    ]

    totals = [0, 0, 0]
    consistent = [0, 0, 0]

    for i in range(len(concept_tuples)):
        group_tuple = concept_tuples[i]        # 3-tuple of (n_worlds, group_size) arrays
        labels_arr = np.asarray(label_sets[i]) # (M, 4)
        if labels_arr.ndim != 2 or labels_arr.shape[0] == 0:
            continue

        for g, (grp_idx, grp_worlds, extract_fn, l_idx) in enumerate(
            zip(group_idx_list, group_tuple, extract_fns, label_slice_idx)
        ):
            # Achievable label bits from concept worlds in this group
            achievable = set()
            for row in grp_worlds:
                eval_vec = np.zeros(21, dtype=int)
                for local_i, gi in enumerate(grp_idx):
                    eval_vec[gi] = row[local_i]
                lv = np.asarray(logic.forward(eval_vec.reshape(1, -1)))
                lv = lv[0] if lv.ndim == 2 else lv.ravel()
                achievable.add(extract_fn(lv))

            # Unique predicted label combos for this group
            ls_slice = labels_arr[:, l_idx]
            unique_combos = np.unique(ls_slice, axis=0)
            for combo in unique_combos:
                totals[g] += 1
                key = (int(combo[0]), int(combo[1])) if g == 0 else int(combo[0])
                if key in achievable:
                    consistent[g] += 1

    names = ["fs", "l", "r"]
    return {
        f"label_consistency_{name}": consistent[g] / totals[g] if totals[g] > 0 else 1.0
        for g, name in enumerate(names)
    }


def conformal_metrics(prediction_tuples, true_labels, ignore_token=-1):
    """
    Compute conformal metrics for concept combinations (tuples).

    Args:
        prediction_tuples: List of np.ndarrays [Samples][Combinations, Concepts],
                           or list of 3-tuples (fs_worlds, l_worlds, r_worlds) for BOIA.
        true_labels: np.ndarray [Samples, Concepts]
    """
    # BOIA group-tuple path: delegate to dedicated function
    if len(prediction_tuples) > 0 and isinstance(prediction_tuples[0], tuple):
        m = boia_conformal_metrics(prediction_tuples, true_labels)
        return m["coverage_avg"], m["set_size_avg"]

    N = len(true_labels)

    coverage_total = 0.0
    set_size_total = 0.0

    for i in range(N):
        sample_tuples = prediction_tuples[i]
        ground_truth = true_labels[i]

        # Flatten extra leading dims (e.g. (1, 4) → (4,) for multilabel)
        if hasattr(ground_truth, "ndim") and ground_truth.ndim > 1:
            ground_truth = ground_truth.reshape(-1)

        assert (
            len(ground_truth.shape) == 1
        ), f"Ground-truth: dim of the worlds. Got {ground_truth.shape, ground_truth.tolist()}"
        assert (
            len(sample_tuples.shape) == 2 or len(sample_tuples) == 0
        ), f"Predictions: must be number of elements in the conformal set, size of the world. Got {sample_tuples.shape, sample_tuples.tolist(), len(sample_tuples)}"

        # Filter tuples: ignore any tuple that is all ignore tokens
        valid_mask = ~np.all(sample_tuples == ignore_token, axis=-1)
        valid_tuples = sample_tuples[valid_mask]

        # BOIA factorized concept sets: each row covers only one group (FS/L/R),
        # with EMPTY_TOKEN for the other groups' concepts.  Coverage holds when
        # every group independently covers the corresponding slice of ground_truth.
        n_concepts = ground_truth.shape[0]
        if n_concepts == 21 and valid_tuples.size > 0 and np.any(valid_tuples == ignore_token):
            BOIA_GROUPS = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8],       # FS
                [9, 10, 11, 18, 19, 20],             # L
                [12, 13, 14, 15, 16, 17],            # R
            ]
            covered = True
            for grp in BOIA_GROUPS:
                grp_rows = valid_tuples[valid_tuples[:, grp[0]] != ignore_token]
                if grp_rows.size == 0:
                    covered = False
                    break
                if not np.any(np.all(ground_truth[grp] == grp_rows[:, grp], axis=-1)):
                    covered = False
                    break
            if covered:
                coverage_total += 1
        else:
            if valid_tuples.size > 0:
                if np.any(np.all(ground_truth == valid_tuples, axis=-1)):
                    coverage_total += 1

        set_size_total += len(valid_tuples)

    # Calculate final averages
    coverage = coverage_total / N
    avg_set_size = set_size_total / N

    return coverage, avg_set_size


def prediction_consistency(concept_tuples, label_sets, logic, EMPTY_TOKEN=-1):
    """
    Consistency metrics for the predicted concept tuples and the predicted label sets.
    """
    # BOIA group-tuple path: delegate to dedicated function
    if len(concept_tuples) > 0 and isinstance(concept_tuples[0], tuple):
        return _prediction_consistency_boia_groups(concept_tuples, label_sets, logic)

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
                if logic.multi_set_logic is None:
                    derived_label = logic.forward(t.reshape(1, -1))
                    derived_label = np.array(derived_label).ravel()
                else:
                    derived_label = logic.forward_multi_set(t.reshape(1, -1))
                    derived_label = np.unique(np.concatenate(derived_label))
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


def _boia_concept_covered(group_tuple, c_star):
    """Check if ground-truth c_star (21-dim) is covered by a BOIA 3-tuple representation."""
    for grp_idx, grp_worlds in zip(
        [_BOIA_FS_IDX, _BOIA_L_IDX, _BOIA_R_IDX], group_tuple
    ):
        c_slice = c_star[grp_idx]
        if len(grp_worlds) == 0 or not np.any(np.all(c_slice == grp_worlds, axis=1)):
            return False
    return True


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

    boia_mode = len(concept_tuples) > 0 and isinstance(concept_tuples[0], tuple)

    for i in range(N):
        tuples_i = concept_tuples[i]  # (K, n_concepts) or 3-tuple for BOIA
        labels_i = label_sets[i]  # (M, 1) or (M,)
        c_star = true_concepts[i]  # (n_concepts,)
        y_star = true_labels[i]

        if boia_mode:
            c_covered = _boia_concept_covered(tuples_i, c_star)
        else:
            c_covered = np.any(np.all(tuples_i == c_star, axis=1))

        y_star_arr = np.asarray(y_star)
        labels_i_arr = np.asarray(labels_i)
        if labels_i_arr.ndim == 2 and y_star_arr.ndim == 1:
            # multilabel: check if y_star row appears in label set rows
            y_covered = bool(np.any(np.all(y_star_arr == labels_i_arr, axis=1)))
        else:
            flat_labels = labels_i_arr.ravel()
            y_covered = bool(np.any(np.isin(y_star_arr, flat_labels)))

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


def joint_failure_metrics(
    concept_tuples,
    label_sets,
    abduced_concept_tuples,
    deduced_label_sets,
    true_concepts,
    true_labels,
):
    """
    Empirical estimates of the joint-failure correction terms from Propositions A.7 and A.8.

    joint_c_miss = Pr(c* ∉ Γβ  ∧  c* ∉ Γab)   — Prop A.7 bound correction
    joint_y_miss = Pr(y* ∉ Υα  ∧  y* ∉ Υde)   — Prop A.8 bound correction

    Args:
        concept_tuples:          Γβ  — conformal concept sets   [N][K, n_concepts]
        label_sets:              Υα  — conformal label sets      [N][M] or [N][M, 1]
        abduced_concept_tuples:  Γab — abduced concept sets      [N][K', n_concepts]
        deduced_label_sets:      Υde — deduced label sets        [N][M'] or [N][M', 1]
        true_concepts:           c*  — ground-truth concepts     [N, n_concepts]
        true_labels:             y*  — ground-truth labels       [N] or [N, 1]
    """
    N = len(concept_tuples)
    c_miss_count = 0
    y_miss_count = 0

    boia_mode = len(concept_tuples) > 0 and isinstance(concept_tuples[0], tuple)

    for i in range(N):
        tuples_i = concept_tuples[i]
        labels_i = label_sets[i]
        ab_tuples_i = abduced_concept_tuples[i]
        de_labels_i = deduced_label_sets[i]
        c_star = true_concepts[i]
        y_star = true_labels[i]

        if boia_mode:
            c_in_beta = _boia_concept_covered(tuples_i,    c_star)
            c_in_ab   = _boia_concept_covered(ab_tuples_i, c_star)
        else:
            c_in_beta = (
                np.any(np.all(tuples_i == c_star, axis=1)) if len(tuples_i) > 0 else False
            )
            c_in_ab = (
                np.any(np.all(ab_tuples_i == c_star, axis=1))
                if len(ab_tuples_i) > 0
                else False
            )

        if not c_in_beta and not c_in_ab:
            c_miss_count += 1

        y_star_arr = np.asarray(y_star)
        labels_i_arr = np.asarray(labels_i)
        de_labels_i_arr = np.asarray(de_labels_i)
        if labels_i_arr.ndim == 2 and y_star_arr.ndim == 1:
            y_in_alpha = bool(np.any(np.all(y_star_arr == labels_i_arr, axis=1)))
            y_in_de = bool(np.any(np.all(y_star_arr == de_labels_i_arr, axis=1)))
        else:
            y_in_alpha = bool(np.any(np.isin(y_star_arr, labels_i_arr.ravel())))
            y_in_de = bool(np.any(np.isin(y_star_arr, de_labels_i_arr.ravel())))

        if not y_in_alpha and not y_in_de:
            y_miss_count += 1

    joint_c_miss = c_miss_count / N
    joint_y_miss = y_miss_count / N
    return joint_c_miss, joint_y_miss


def boia_conditional_conformal_metrics_per_group(
    concept_tuples, label_sets, true_concepts, true_labels
):
    """
    Per-group δab and δde for BOIA factorized representation.

    δab_g = P(c*_g ∈ Γab_g | y*_g ∈ Υα_g)
    δde_g = P(y*_g ∈ Υde_g | c*_g ∈ Γβ_g)

    concept_tuples: list of 3-tuples (fs_worlds, l_worlds, r_worlds) per sample
    label_sets:     list of (M, 4) label arrays per sample
    true_concepts:  (N, 21) ground-truth concept array
    true_labels:    (N, 4)  ground-truth label array
    """
    N = len(concept_tuples)
    group_concept_idx = [_BOIA_FS_IDX, _BOIA_L_IDX, _BOIA_R_IDX]
    group_label_idx   = [_BOIA_LABEL_FS_IDX, _BOIA_LABEL_L_IDX, _BOIA_LABEL_R_IDX]

    ab_num = [0, 0, 0]
    ab_den = [0, 0, 0]
    de_num = [0, 0, 0]
    de_den = [0, 0, 0]

    for i in range(N):
        group_worlds = list(concept_tuples[i])      # [fs_worlds, l_worlds, r_worlds]
        ls = np.asarray(label_sets[i])              # (M, 4)
        c_star = np.asarray(true_concepts[i]).ravel()
        y_star = np.asarray(true_labels[i]).ravel()

        for g, (c_idx, l_idx) in enumerate(zip(group_concept_idx, group_label_idx)):
            c_star_g = c_star[c_idx]
            y_star_g = y_star[l_idx]
            grp_worlds = group_worlds[g]

            c_covered = (
                len(grp_worlds) > 0
                and bool(np.any(np.all(c_star_g == grp_worlds, axis=1)))
            )
            y_covered = (
                ls.ndim == 2
                and ls.shape[0] > 0
                and bool(np.any(np.all(y_star_g == ls[:, l_idx], axis=1)))
            )

            if y_covered:
                ab_den[g] += 1
                if c_covered:
                    ab_num[g] += 1
            if c_covered:
                de_den[g] += 1
                if y_covered:
                    de_num[g] += 1

    names = ["fs", "l", "r"]
    result = {}
    for g, name in enumerate(names):
        result[f"delta_ab_{name}"] = (
            ab_num[g] / ab_den[g] if ab_den[g] > 0 else float("nan")
        )
        result[f"delta_de_{name}"] = (
            de_num[g] / de_den[g] if de_den[g] > 0 else float("nan")
        )
    return result


def boia_joint_failure_metrics_per_group(
    concept_beta_tuples,
    label_alpha_sets,
    concept_ab_tuples,
    label_de_sets,
    true_concepts,
    true_labels,
):
    """
    Per-group joint-failure correction terms for BOIA (Propositions A.7 and A.8).

    joint_c_g = Pr(c*_g ∉ Γβ_g ∧ c*_g ∉ Γab_g)
    joint_y_g = Pr(y*_g ∉ Υα_g ∧ y*_g ∉ Υde_g)

    concept_beta_tuples: Γβ  (CO concept sets),  list of 3-tuples
    label_alpha_sets:    Υα  (TaskOnly label sets), list of (M, 4) arrays
    concept_ab_tuples:   Γab (TA concept sets),  list of 3-tuples
    label_de_sets:       Υde (CDe label sets),   list of (M, 4) arrays
    true_concepts:       (N, 21)
    true_labels:         (N, 4)
    """
    N = len(concept_beta_tuples)
    group_concept_idx = [_BOIA_FS_IDX, _BOIA_L_IDX, _BOIA_R_IDX]
    group_label_idx   = [_BOIA_LABEL_FS_IDX, _BOIA_LABEL_L_IDX, _BOIA_LABEL_R_IDX]

    c_miss = [0, 0, 0]
    y_miss = [0, 0, 0]

    for i in range(N):
        beta_groups = list(concept_beta_tuples[i])
        ab_groups   = list(concept_ab_tuples[i])
        la = np.asarray(label_alpha_sets[i])   # Υα, (M, 4)
        ld = np.asarray(label_de_sets[i])       # Υde, (M, 4)
        c_star = np.asarray(true_concepts[i]).ravel()
        y_star = np.asarray(true_labels[i]).ravel()

        for g, (c_idx, l_idx) in enumerate(zip(group_concept_idx, group_label_idx)):
            c_star_g = c_star[c_idx]
            y_star_g = y_star[l_idx]

            c_in_beta = (
                len(beta_groups[g]) > 0
                and bool(np.any(np.all(c_star_g == beta_groups[g], axis=1)))
            )
            c_in_ab = (
                len(ab_groups[g]) > 0
                and bool(np.any(np.all(c_star_g == ab_groups[g], axis=1)))
            )
            if not c_in_beta and not c_in_ab:
                c_miss[g] += 1

            y_in_alpha = (
                la.ndim == 2 and la.shape[0] > 0
                and bool(np.any(np.all(y_star_g == la[:, l_idx], axis=1)))
            )
            y_in_de = (
                ld.ndim == 2 and ld.shape[0] > 0
                and bool(np.any(np.all(y_star_g == ld[:, l_idx], axis=1)))
            )
            if not y_in_alpha and not y_in_de:
                y_miss[g] += 1

    names = ["fs", "l", "r"]
    result = {}
    for g, name in enumerate(names):
        result[f"joint_c_{name}"] = c_miss[g] / N
        result[f"joint_y_{name}"] = y_miss[g] / N
    return result
