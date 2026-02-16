import re
import pandas as pd
from pathlib import Path

from conformal.general_utils import log
from conformal.datasets import boia, chx, derma, mnistadd, mnisthalf, mnistsump, mnistaddn


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
        seed = extract_seed(Path(f).name, folder_name, seed_position)

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
    return f[seed_position]


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
            metric_name = row['metric']
            value = row['value']

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


def generate_latex_table(rows, caption="Results Summary"):
    """
    Generate LaTeX table with mean \pm std per metric across seeds and settings.
    Args:
        rows: list of dicts, each with keys: 'seed', 'setting', 'metric', 'value'
        caption: caption for LaTeX table
    Returns:
        str: LaTeX table code
    """
    df = pd.DataFrame(rows)
    df['seed'] = df['seed'].astype(int)

    agg_df = df.groupby(['setting', 'metric'])['value'].agg(['mean', 'std']).reset_index()
    agg_df['mean_std'] = agg_df.apply(lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1)
    table_df = agg_df.pivot(index='metric', columns='setting', values='mean_std')
    table_df = table_df.sort_index()
    latex_table = table_df.to_latex(
        escape=False,
        caption=caption,
        label="tab:results",
        na_rep="-",
        column_format="l" + "c" * len(table_df.columns),
    )
    return latex_table


def main(experiment_name, results_output_h, stats_output_h, args, device):
    """Main function that parses the arguments and writes the output."""
    files = collect_result_files(args.output_dir_path, build_seed_regex(args))
    loaded = load_results(files, str(args.output_dir_path), get_seed_position(args))

    all_rows = []
    for seed, data in loaded:
        all_rows.extend(flatten_results(seed, data))
    latex_table = generate_latex_table(all_rows)

    original_path = Path(stats_output_h.name)
    results_stats_path = original_path.with_name(
        f"{original_path.stem}.tex"
    )

    log(f"LATEX:\n{latex_table}", "INFO")

    log(f"Writing table of results to {results_stats_path}...", "INFO")
    with open(results_stats_path, "w") as f:
        f.write(f"{latex_table}")
    log("Results written successfully.", "INFO")


if __name__ == "__main__":
    main()