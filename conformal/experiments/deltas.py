import csv
from pathlib import Path

from conformal.general_utils import log
from conformal.datasets import (
    boia,
    chx,
    mnistadd,
    mnisthalf,
    mnistsump,
    mnistaddn,
    mnistevenodd,
    rival,
    cifar,
    cebab,
)

from conformal.experiments.analyze import (
    collect_result_files,
    build_seed_regex,
    get_seed_position,
    load_results,
    flatten_results,
    extract_seed,
)


def configure_global_arguments(parser):

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
    parser.add_argument(
        "--epsilon-symbols",
        type=float,
        default=0.2807344052335263,
        help="DSL hyperparameter for learning symbols.",
    )
    parser.add_argument(
        "--epsilon-rules",
        type=float,
        default=0.1077119516324264,
        help="DSL hyperparameter for learning rules.",
    )


def test_parser(parser):
    configure_global_arguments(parser)

    subparsers = parser.add_subparsers(dest="dataset")
    mnistadd.configure_subparsers(subparsers)
    mnisthalf.configure_subparsers(subparsers)
    mnistsump.configure_subparsers(subparsers)
    mnistaddn.configure_subparsers(subparsers)
    chx.configure_subparsers(subparsers)
    boia.configure_subparsers(subparsers)
    rival.configure_subparsers(subparsers)
    cifar.configure_subparsers(subparsers)
    cebab.configure_subparsers(subparsers)
    mnistevenodd.configure_subparsers(subparsers)


def configure_subparsers(subparsers):
    parser = subparsers.add_parser(
        "deltas",
        help="Analyze the deltas in the results",
    )
    test_parser(parser)
    parser.set_defaults(func=main)


def main(experiment_name, results_output_h, stats_output_h, args, device):
    files = collect_result_files(args.output_dir_path, build_seed_regex(args))
    seed_position = get_seed_position(args)
    loaded = load_results(files, str(args.output_dir_path), seed_position)

    if len(loaded) == 0:
        log("No results found to analyze. Exiting.", "WARNING")
        return

    all_rows = []
    for seed, data in loaded:
        all_rows.extend(flatten_results(seed, data))

    log(f"Loaded results from {len(loaded)} seeds.", "INFO")

    average_deltas = {}
    for row in all_rows:
        if "delta_ab" in row["metric"]:
            average_deltas.setdefault("delta_ab", []).append(row["value"])
        if "delta_de" in row["metric"]:
            average_deltas.setdefault("delta_de", []).append(row["value"])

    summary = {}
    for delta_name, values in average_deltas.items():
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        summary[delta_name] = {"mean": mean, "std": std, "n": len(values)}
        log(
            f"Average {delta_name}: {mean:.4f} ± {std:.4f} with {len(values)} samples",
            "INFO",
        )

    splitted_path, _ = extract_seed(
        stats_output_h.name.split("/")[-1], str(args.output_dir_path), seed_position
    )
    splitted_path[seed_position] = f"{len(loaded)}_seeds"
    splitted_path = "_".join(splitted_path)
    splitted_path = Path(args.output_dir_path, splitted_path)
    delta_path = splitted_path.with_name(f"{splitted_path.stem}_deltas.csv")

    log(f"Writing table of results to {delta_path}...", "INFO")

    with open(delta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [f"{name}_mean" for name in summary] + [f"{name}_std" for name in summary]
        )
        writer.writerow(
            [v["mean"] for v in summary.values()] + [v["std"] for v in summary.values()]
        )

    log("Results written successfully.", "INFO")


if __name__ == "__main__":
    main()