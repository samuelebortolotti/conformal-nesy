import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import math


def save_line_plot(
    data,
    output_path,
    labels=None,
    xlabel="Step",
    ylabel="Value",
    title="Plot",
    figsize=(10, 6),
    legend=True,
    colors=None,
    style="-",
    grid=True,
    transparent=False,
):
    """
    Save a clean, well-formatted line plot with multiple series.

    Args:
        data (array-like): Either a 1D array or a 2D array of shape (steps, n_lines)
        output_path (str): Path to save the figure
        labels (list of str, optional): Labels for each line
        xlabel, ylabel, title (str): Axis labels and plot title
        figsize (tuple): Figure size in inches
        legend (bool): Whether to show the legend
        colors (list of str, optional): List of colors for lines
        style (str): Line style (e.g. '-', '--', ':')
        grid (bool): Whether to display gridlines
        transparent (bool): Save figure with transparent background
    """
    plt.figure(figsize=figsize)
    plt.style.use("default")

    # Convert to numpy array if list
    if isinstance(data, list):
        data = (
            np.stack(data, axis=1)
            if isinstance(data[0], (list, np.ndarray))
            else np.array(data)
        )

    data = np.array(data)
    n_lines = data.shape[1] if data.ndim > 1 else 1
    steps = np.arange(data.shape[0])

    # Plot each line
    for i in range(n_lines):
        y = data[:, i] if n_lines > 1 else data
        label = labels[i] if labels is not None and i < len(labels) else f"Line {i+1}"
        color = colors[i] if colors is not None and i < len(colors) else None
        plt.plot(steps, y, style, label=label, color=color, linewidth=2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(bottom=0, top=1)
    plt.title(title, fontsize=14, fontweight="bold")
    if grid:
        plt.grid(True, linestyle="--", alpha=0.6)
    if legend:
        plt.legend(frameon=False, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight", transparent=transparent)
    plt.close()


def plot_confusion_matrix(
    y_true, y_pred, class_names, title, output_path=None, max_cols=4, multilabel=False
):
    """
    Plot multiple confusion matrices (one per label) in a grid.

    Args:
        y_true (ndarray): shape (n_samples, n_labels)
        y_pred (ndarray): shape (n_samples, n_labels)
        class_names (list of str): list of label names (length n_labels)
        title (str): figure title
        output_path (str, optional): path to save the PDF
        max_cols (int): maximum number of confusion matrices per row
    """
    if y_true.ndim == 1:
        n_labels = 1
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    else:
        n_labels = y_true.shape[1]

    if multilabel:
        n_labels = y_true.shape[1]
        n_cols = min(max_cols, n_labels)
        n_rows = math.ceil(n_labels / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array([axes]).flatten() if n_labels == 1 else axes.flatten()

        for i in range(n_labels):
            # Binary comparison for each attribute
            cm = confusion_matrix(
                y_true[:, i], y_pred[:, i], labels=[0, 1], normalize="true"
            )
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f",
                cmap="Reds",
                ax=axes[i],
                cbar=False,
                xticklabels=["Abs", "Pres"],
                yticklabels=["Abs", "Pres"],
            )
            axes[i].set_title(class_names[i] if i < len(class_names) else f"Attr {i}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")

        # Clean up empty subplots
        for j in range(n_labels, len(axes)):
            axes[j].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred, normalize="true")

        # Limit labels for readability if there are too many (like 50)
        tick_labels = class_names if len(class_names) < 20 else False

        sns.heatmap(
            cm,
            annot=(len(class_names) < 20),
            fmt=".1f",
            cmap="Reds",
            ax=ax,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )
        ax.set_title("Multi-class Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        plt.savefig(output_path, format="pdf")

    plt.close(fig)


def plot_knowledge(concept_preds, labels, output_path, title):
    assert concept_preds.shape[1] == 2, "Expected two concept predictions per sample"
    max_per_column = concept_preds.max(axis=0) + 1
    heatmap = np.full((max_per_column[0], max_per_column[1]), np.nan)

    for cp, label in zip(concept_preds, labels):
        i, j = cp
        if heatmap[i, j] != heatmap[i, j]:
            heatmap[i, j] = 0
        heatmap[i, j] = label

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap + 10, cmap="Reds")

    for i in range(max_per_column[0]):
        for j in range(max_per_column[1]):
            val = heatmap[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="w", fontsize=8)

    ax.set_xlabel("Concept 2")
    ax.set_ylabel("Concept 1")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Avg Label")
    plt.tight_layout()

    plt.savefig(output_path, format="pdf")
    plt.close()


def entropy_plot_over_time(
    entropy_values, concept_names, output_path, title="Concept Entropy"
):
    """
    Plots the average entropy for each concept as a bar plot and saves the figure.

    Parameters:
    - entropy_values: np.ndarray of shape (N, V), entropy values per concept
    - concept_names: list of length V, names of the concepts
    - output_path: str, path to save the figure (PDF)
    - title: str, title of the plot
    """
    entropy_values = np.asarray(entropy_values)
    assert entropy_values.ndim == 2, "Input should be 2D: (N, V)"
    N, V = entropy_values.shape
    assert (
        len(concept_names) == V
    ), "Length of concept_names must match number of concepts"

    timesteps = np.arange(N)

    fig, ax = plt.subplots(figsize=(10, 6))
    for v in range(V):
        ax.plot(timesteps, entropy_values[:, v], label=concept_names[v], linewidth=1.8)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Entropy")
    ax.set_title(title)
    ax.legend(title="Concepts", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(output_path, format="pdf")
    plt.close()


def plot_conformal_comparison(results_storage, file_name, target_coverage=0.9):
    """
    Scatter plot comparing Label Coverage vs Average Set Size for different methods.

    Args:
        results_storage (dict): Dictionary containing metrics for each method.
        file_name (Path): Path to save the image.
        target_coverage (float): The desired coverage level (1 - alpha).
    """

    markers = ['o', 's', '^', 'D', 'p', '*', 'X']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    plotted_methods = []

    for name, coverage, coverage_size in zip(
        ["concepts", "labels"],
        ["coverage_concepts", "coverage_labels"],
        ["concept_size", "label_size"],
    ):
        plt.figure(figsize=(10, 7))

        for i, (method_name, metrics) in enumerate(results_storage.items()):
            if name == "labels" and "coverage_labels" not in metrics:
                continue
            if name == "concepts" and "coverage_concepts" not in metrics:
                continue

            x = metrics[coverage]
            y = metrics[coverage_size]

            plt.scatter(
                x,
                y,
                label=method_name,
                s=150,
                alpha=0.9,
                edgecolors="black",
                linewidth=1.5,
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
            )
            plotted_methods.append(method_name)

        plt.axvline(
            x=target_coverage,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Target ({target_coverage})",
        )
        plt.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        plt.title(
            f"{name.title()} Conformal Prediction: Coverage vs. Size", fontsize=14
        )
        plt.xlabel("Marginal Coverage (Higher is better)", fontsize=12)
        plt.ylabel("Average Set Size (Lower is better, Ideal=1)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(loc="best", frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig(f"{file_name}.conformal_comparison_{name}.pdf", dpi=300)
        plt.close()


def plot_consistency_comparison(results_storage, file_name, target_coverage=0.9):
    """
    Scatter plot comparing Concept and Label Consistency 

    Args:
        results_storage (dict): Dictionary containing metrics for each method.
        file_name (Path): Path to save the image.
        target_coverage (float): The desired coverage level (1 - alpha).
    """

    markers = ['o', 's', '^', 'D', 'p', '*', 'X']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    plt.figure(figsize=(10, 7))

    for i, (method_name, metrics) in enumerate(results_storage.items()):
        if "concept_consistency" not in metrics or "label_consistency" not in metrics:
            continue

        x = metrics["concept_consistency"]
        y = metrics["label_consistency"]

        plt.scatter(
            x,
            y,
            label=method_name,
            s=150,
            alpha=0.9,
            edgecolors="black",
            linewidth=1.5,
            marker=markers[i % len(markers)],
            c=colors[i % len(colors)],
        )

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7, label="Perfect Consistency")

    plt.title("Prediction Consistency: Concepts vs Labels", fontsize=14)
    plt.xlabel("Concept Consistency (Higher is better)", fontsize=12)
    plt.ylabel("Label Consistency (Higher is better)", fontsize=12)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f"{file_name}.consistency_comparison.pdf", dpi=300)
    plt.close()


def plot_model_metrics(
    baseline_results, file_name, concept_names=None, multiconcepts=False
):
    """
    Plots scalar performance metrics (F1, Loss, ECE) and per-concept entropy.

    Args:
        baseline_results (dict): The dictionary stored under results_storage['No Conformal']
        file_name (Path): File name to save plots.
        concept_names (list): Optional list of names for the concepts.
    """

    f1_metrics = {
        "Label F1": baseline_results.get("test_f1", 0),
        "Concept F1": baseline_results.get("test_c_f1", 0),
    }

    plt.figure(figsize=(6, 5))
    plt.bar(
        f1_metrics.keys(),
        f1_metrics.values(),
        color=["#2ca02c", "#1f77b4"],
        alpha=0.8,
        width=0.5,
    )
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Classification Performance (F1)")
    plt.tight_layout()
    plt.savefig(f"{file_name}.metric_f1_scores.pdf", dpi=300)
    plt.close()

    unc_metrics = {
        "Test Loss": baseline_results.get("test_loss", 0),
        "Label ECE": baseline_results.get("yece", 0),
        "Concept ECE": baseline_results.get("cece", 0),
        "Mean Entropy (H_c)": baseline_results.get("H_c", 0),
    }

    plt.figure(figsize=(8, 5))
    plt.bar(
        unc_metrics.keys(), unc_metrics.values(), color="#d62728", alpha=0.7, width=0.6
    )
    plt.ylabel("Value")
    plt.title("Calibration & Uncertainty Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{file_name}.metric_uncertainty.pdf", dpi=300)
    plt.close()

    h_per_val = np.array(baseline_results.get("H_c_per_value", []))

    if h_per_val.size > 0:
        plt.figure(figsize=(12, 6))

        # If multiconcepts is True, we handle the [N_concepts, N_classes] structure
        if multiconcepts:
            # Squeeze if it's (1, 21, 2) -> (21, 2)
            if h_per_val.ndim == 3:
                h_per_val = h_per_val.squeeze(0)

            n_concepts, n_classes = h_per_val.shape
            indices = np.arange(n_concepts)
            width = 0.35  # Width of individual bars

            # Plot bars for each class (e.g., False and True)
            # Assuming class 0 is False/Absent and class 1 is True/Present
            plt.bar(
                indices - width / 2,
                h_per_val[:, 0],
                width,
                label="Class 0 (False)",
                color="orchid",
                alpha=0.7,
            )
            plt.bar(
                indices + width / 2,
                h_per_val[:, 1],
                width,
                label="Class 1 (True)",
                color="indigo",
                alpha=0.7,
            )

            plt.legend()
        else:
            # Standard 1D entropy per concept logic
            n_concepts = len(h_per_val)
            indices = np.arange(n_concepts)
            plt.bar(indices, h_per_val, color="purple", alpha=0.6, edgecolor="black")

        # Labels and Formatting
        if concept_names and len(concept_names) == n_concepts:
            labels = concept_names
        else:
            labels = [f"C{i}" for i in range(n_concepts)]

        plt.xlabel("Concepts")
        plt.ylabel("Entropy (Uncertainty)")
        plt.title("Per-Concept/State Uncertainty (Entropy)")
        plt.xticks(indices, labels, rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.ylim(0, max(h_per_val.max() * 1.1, 1.0))  # Dynamic limit

        plt.tight_layout()
        plt.savefig(f"{file_name}.metric_entropy_per_concept.pdf", dpi=300)
        plt.close()
