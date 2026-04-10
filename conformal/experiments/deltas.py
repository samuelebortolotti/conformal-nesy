import csv
from pathlib import Path

from conformal.general_utils import log
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

from conformal.experiments.analyze import (
    collect_result_files,
    build_seed_regex,
    get_seed_position,
    load_results,
    flatten_results,
    extract_seed,
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
    rival.configure_subparsers(subparsers)
    cifar.configure_subparsers(subparsers)
    cebab.configure_subparsers(subparsers)
    mnistevenodd.configure_subparsers(subparsers)


def configure_subparsers(subparsers):
    """Configure the subparsers."""
    parser = subparsers.add_parser(
        "deltas",
        help="Analyze the deltas in the results",
    )
    test_parser(parser)
    parser.set_defaults(func=main)


def main(experiment_name, results_output_h, stats_output_h, args, device):
    """Main function that parses the arguments and writes the output."""
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
            delta_ab = row["value"]
            average_deltas.setdefault("delta_ab", []).append(delta_ab)
        if "delta_de" in row["metric"]:
            delta_de = row["value"]
            average_deltas.setdefault("delta_de", []).append(delta_de)

    for delta_name, values in average_deltas.items():
        average_value = sum(values) / len(values)
        average_deltas[delta_name] = average_value
        log(
            f"Average {delta_name}: {average_value:.4f} with {len(values)} samples",
            "INFO",
        )

    # Change the name with the number of seeds analyzed
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
        writer.writerow(average_deltas.keys())
        writer.writerow(average_deltas.values())

    log("Results written successfully.", "INFO")


if __name__ == "__main__":
    main()
