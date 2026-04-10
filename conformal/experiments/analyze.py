import re
import pandas as pd
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
        "analyze",
        help="Analyze the results coming from test",
    )
    test_parser(parser)
    parser.set_defaults(func=main)


def collect_result_files(root_dir, seed_regex):
    """Get the csv files"""
    log(f"Scanning folder {root_dir} for test result files...", "INFO")
    pattern = re.compile(seed_regex + ".train_results_conformal.csv$")
    files = list(Path(root_dir).rglob("*.train_results_conformal.csv"))
    matched_files = [f for f in files if pattern.match(f.name)]
    log(f"Found {len(matched_files)} result files.", "INFO")
    for f in matched_files:
        log(f"Found: {f}", "INFO")
    return matched_files


def load_results(files, folder_name, seed_position):
    """
    Load all CSV results, split by setting, attach seed.
    Returns:
        List of tuples: (seed, dict_of_settings)
        where dict_of_settings: {setting_name: pd.DataFrame(metrics)}
    """
    all_results = []

    for f in files:
        log(f"Loading {f}", "DEBUG")

        # Extract seed
        _, seed = extract_seed(Path(f).name, folder_name, seed_position)

        settings_dict = {}
        current_setting = None
        metrics = {}

        with open(f, "r") as csvfile:
            for line in csvfile:
                line = line.strip()
                if not line or line.startswith("H_c_per_value"):
                    continue
                if line.startswith("# Results for setting:"):
                    if current_setting is not None:
                        settings_dict[current_setting] = pd.DataFrame(
                            list(metrics.items()), columns=["metric", "value"]
                        )
                    current_setting = line.replace("# Results for setting:", "").strip()
                    metrics = {}
                else:
                    if "," not in line:
                        continue
                    key, val = line.split(",", 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        val_num = float(val)
                        metrics[key] = val_num
                    except ValueError:
                        metrics[key] = val
            if current_setting is not None:
                settings_dict[current_setting] = pd.DataFrame(
                    list(metrics.items()), columns=["metric", "value"]
                )

        all_results.append((seed, settings_dict))

    log(f"Loaded {len(all_results)} result files.", "INFO")
    return all_results


def build_seed_regex(args):
    """
    Build a regex that matches get_basename(args) but with any seed.
    """
    keys_ordered = [k for k in vars(args) if k not in ["func", "model_path"]]

    regex_parts = []
    for k in keys_ordered:
        v = str(vars(args)[k])
        if k == "seed":
            regex_parts.append(r"(\d+)")  # replace seed with capture group
        else:
            regex_parts.append(re.escape(v))  # escape special chars

    pattern = "_".join(regex_parts)
    pattern = "^experiment_" + pattern
    return pattern


def extract_seed(f, folder_name, seed_position):
    """Extract the seed from the file"""
    f = f.replace(folder_name + "_", "")
    f = f.split("_")
    return f, f[seed_position]


def get_seed_position(args):
    """Get the seed position from the args"""
    args_dict = vars(args)
    ignore_keys = ["func", "model_path", "analyze_seed"]
    keys_ordered = [k for k in args_dict.keys() if k not in ignore_keys]
    return keys_ordered.index("seed")


def flatten_results(seed, results_dict):
    """Get seed, settings, metrics and values for aggregation"""
    rows = []
    for setting, df_metrics in results_dict.items():
        for _, row in df_metrics.iterrows():
            metric_name = row["metric"]
            value = row["value"]

            if isinstance(value, (int, float)):
                rows.append(
                    {
                        "seed": seed,
                        "setting": setting,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
    return rows


def beautify_method_names(table_df, nesy_method):
    row_map = {
        "No Conformal": nesy_method.upper(),
        "Conformal Concepts Only": "ConceptsOnly",
        "Conformal Hard Logic": "ConceptsPlusDeduction",
        "Conformal both Concepts and Labels": "CL",
        "Conformal with Abduction": "TaskPlusAbduction",
        "Conformal only Labels": "TaskOnly",
        "Conformal both Concepts and Labels with Concept Refinement": "JCC",
        "Conformal both Concepts and Labels with Label Refinement": "bastani",
        "Conformal both Concepts and Labels with Concept and Label Refinement": "method",
    }

    return table_df.rename(index=row_map)


def split_table(table_df):
    nesy_metrics = [
        m
        for m in table_df.index
        if m in ["H\_c", "cece", "yece"] or m.startswith("test")
    ]
    nesy_table_df = table_df.loc[nesy_metrics, ["NeSy"]]
    nesy_table_latex = nesy_table_df.to_latex(
        escape=False,
        na_rep="-",
        column_format="l" + "c",
    )

    other_metrics = [m for m in table_df.index if m not in nesy_metrics]
    conformal_table_df = table_df.loc[other_metrics]
    conformal_table_latex = conformal_table_df.to_latex(
        escape=False,
        na_rep="-",
        column_format="l" + "c" * len(conformal_table_df.columns),
    )

    return conformal_table_latex, nesy_table_latex


def generate_latex_table(rows, nesy_method):

    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype(int)

    agg_df = (
        df.groupby(["setting", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg_df["mean_std"] = agg_df.apply(
        lambda x: f"${x['mean']:5.3f} \\pm {x['std']:5.3f}$", axis=1
    )

    table_df = agg_df.pivot(index="setting", columns="metric", values="mean_std")

    table_df = beautify_method_names(table_df, nesy_method)

    nesy_name = nesy_method.upper()

    desired_order = [
        nesy_name,
        "TaskOnly",
        "TaskPlusAbduction",
        "ConceptsOnly",
        "ConceptsPlusDeduction",
        "bastani",
        "method"
    ]

    table_df = table_df.reindex(desired_order)

    concept_metrics = [
        "concept_consistency",
        "concept_size",
        "coverage_concepts",
    ]

    label_metrics = [
        "label_consistency",
        "label_size",
        "coverage_labels",
    ]

    combined = table_df[concept_metrics + label_metrics]

    combined.columns = pd.MultiIndex.from_tuples(
        [
            ("{\\sc Concepts}", "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} {\\sc Method} & {\\sc Consistency}"),
            ("{\\sc Concepts}", "{\\sc Size}"),
            ("{\\sc Concepts}", "{\\sc Coverage}"),
            ("{\\sc Labels}", "{\\sc Consistency}"),
            ("{\\sc Labels}", "{\\sc Size}"),
            ("{\\sc Labels}", "{\\sc Coverage}"),
        ]
    )

    latex_df = combined.copy()

    latex_index = []
    for name in latex_df.index:
        if name == nesy_name:
            latex_index.append(f"\\{nesy_name}")
        else:
            latex_index.append(f"\\{nesy_name} + \\{name}")

    latex_df.index = latex_index
    
    conformal_latex = latex_df.to_latex(
        escape=False,
        na_rep="-",
        column_format="lcccccc",
        multicolumn=True,
        multicolumn_format="c",
    )

    conformal_latex = conformal_latex.replace(
        "\\begin{tabular}",
        "\\scalebox{0.8}{\n\\begin{tabular}"
    )

    conformal_latex = conformal_latex.replace(
        "\\end{tabular}",
        "\\end{tabular}\n}"
    )

    conformal_latex = conformal_latex.replace(
        "& \\cmidrule(lr){2-4}",
        "\\cmidrule(lr){2-4}"
    )

    nesy_table = table_df.loc[[nesy_name]].copy()
    nesy_table = nesy_table.rename(index={nesy_name: f"\\{nesy_name}"})

    nesy_latex = nesy_table.to_latex(
        escape=False,
        na_rep="-",
        column_format="l" + "c" * len(nesy_table.columns),
    )

    return conformal_latex, nesy_latex


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

    conformal_table, nesy_table = generate_latex_table(
        all_rows, args.nesy
    )

    # Change the name with the number of seeds analyzed
    splitted_path, _ = extract_seed(
        stats_output_h.name.split("/")[-1], str(args.output_dir_path), seed_position
    )
    splitted_path[seed_position] = f"{len(loaded)}_seeds"
    splitted_path = "_".join(splitted_path)
    splitted_path = Path(args.output_dir_path, splitted_path)
    conforma_results_stats_path = splitted_path.with_name(
        f"{splitted_path.stem}_conformal.tex"
    )
    nesy_results_stats_path = splitted_path.with_name(f"{splitted_path.stem}_nesy.tex")

    log(f"CONFORMAL:\n{conformal_table}", "INFO")
    log(f"NESY:\n{nesy_table}", "INFO")

    log(
        f"Writing table of results to {conforma_results_stats_path}...", "INFO"
    )
    with open(conforma_results_stats_path, "w") as f:
        f.write(f"{conformal_table}")

    log(f"Writing table of results to {nesy_results_stats_path}...", "INFO")
    with open(nesy_results_stats_path, "w") as f:
        f.write(f"{nesy_table}")

    log("Results written successfully.", "INFO")


if __name__ == "__main__":
    main()
