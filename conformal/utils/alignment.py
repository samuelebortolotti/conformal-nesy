import numpy as np
from scipy.optimize import linear_sum_assignment


def align_concepts(probs, labels, multiclass=False):
    """
    Wrapper that applies Hungarian alignment to concept probabilities.
    - If probs is (N, C): directly aligns.
    - If probs is (N, M, C): flattens, aligns, reshapes back.
    """

    if probs.ndim == 2:
        # Shape (N, C) — just apply alignment directly
        aligned_probs, permutation = align_distribution_with_labels(
            probs, labels, multiclass
        )

    elif probs.ndim == 3:

        if not multiclass:
            # Shape (N, M, C) — flatten first two dims
            N, M, C = probs.shape
            flat_probs = probs.reshape(-1, C)
            flat_labels = labels.reshape(-1)

            aligned_flat, permutation = align_distribution_with_labels(
                flat_probs, flat_labels, multiclass
            )

            # Reshape back to (N, M, C)
            aligned_probs = aligned_flat.reshape(N, M, C)
        else:
            aligned_probs, permutation = align_distribution_with_labels(
                probs, labels, multiclass
            )

    else:
        raise ValueError("probs must be either (N, C) or (N, M, C)")

    return aligned_probs, permutation


def align_distribution_with_labels(probs, labels, multiclass=False):

    if not multiclass:

        n_samples, n_classes = probs.shape
        assert labels.shape[0] == n_samples, "Labels and probs size mismatch"

        # One-hot encode labels
        labels_onehot = np.eye(n_classes)[labels]  # shape (N, C)

        # Soft contingency: each entry is sum of probs[i,c_pred] for true class c_true
        contingency = probs.T @ labels_onehot  # shape (C, C)

        # Hungarian algorithm maximizes total matches → minimize negative contingency
        row_ind, col_ind = linear_sum_assignment(-contingency)

        # Build permutation matrix
        P = np.zeros((n_classes, n_classes), dtype=int)
        P[row_ind, col_ind] = 1

        # Apply permutation
        aligned_probs = probs @ P

    else:

        B, N, C = probs.shape
        assert C == 2, "Expect binary class in last dimension"

        # Step 1: predicted labels
        pred_labels = probs.argmax(axis=2).astype(int)  # shape (B, N), 0 or

        # Step 2: contingency matrix along N
        contingency = np.zeros((N, N), dtype=int)
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    # Count matches where both predicted and label are 1
                    if pred_labels[b, i] == 1 and labels[b, j] == 1:
                        contingency[i, j] += 1

        # Step 3: Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-contingency)
        P = np.zeros((N, N), dtype=int)
        P[row_ind, col_ind] = 1

        # Step 4: Apply permutation along N
        aligned_probs = np.zeros_like(probs)
        for b in range(B):
            aligned_probs[b] = P @ probs[b]

    return aligned_probs, P


def inverse_permutation(perm):
    """Return inverse permutation as np.ndarray."""
    return perm.T


def apply_knowledge_permutation(probs, perm):
    """
    Revert previously aligned probabilities using inverse permutation.
    """
    probs = probs @ inverse_permutation(perm)
    return probs


def align_knowledge_input(probs, perm):

    if probs.ndim == 2:
        aligned_probs = apply_knowledge_permutation(probs, perm)
    elif probs.ndim == 3:
        N, M, C = probs.shape
        probs = probs.reshape(-1, C)

        aligned_probs = apply_knowledge_permutation(probs, perm)
        aligned_probs = aligned_probs.reshape(N, M, C)
    else:
        raise ValueError("probs must be either (N, C) or (N, M, C)")

    return aligned_probs
