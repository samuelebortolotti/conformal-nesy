"""Main module that parses command line arguments."""

import argparse
import os
import pathlib
import io
import gzip
import subprocess
import logging

from typing import Optional

import conformal.general_utils as utils
from conformal.experiments import train, optimize, test, analyze


def compressor_7z(file_path: str):
    """ "Return a file-object that compresses data written using 7z."""
    p = subprocess.Popen(
        ["7z", "a", "-si", file_path],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    return io.TextIOWrapper(p.stdin, encoding="utf-8")


def output_writer(path: str, compression: Optional[str]):
    """Write data to a compressed file."""
    if compression == "7z":
        return compressor_7z(path + ".7z")
    elif compression == "gzip":
        return gzip.open(path + ".gz", "wt", encoding="utf-8")
    else:
        return open(path, "wt", encoding="utf-8")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="conformal",
        description="Conformal NeSy methods experiments.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        choices={"cuda", "cpu"},
        default="cuda",
        help="Which device to run",
    )
    parser.add_argument(
        "output_dir_path",
        metavar="OUTPUT_DIR",
        type=pathlib.Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--output-compression",
        choices={None, "7z", "gzip"},
        required=False,
        default=None,
        help="Output compression format.",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        default=False,
        help="Don't write any file",
    )
    parser.add_argument(
        "--entangled",
        "-ent",
        default=False,
        action="store_true",
        help="Entangled model",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=32,
        help="Set the seed for the experiment",
    )

    subparsers = parser.add_subparsers(help="sub-commands help")
    train.configure_subparsers(subparsers)
    optimize.configure_subparsers(subparsers)
    test.configure_subparsers(subparsers)
    analyze.configure_subparsers(subparsers)

    parsed_args = parser.parse_args()
    if "func" not in parsed_args:
        parser.print_usage()
        parser.exit(1)

    return parsed_args


def main():
    """Main function."""
    args = get_args()

    if not args.output_dir_path.exists():
        args.output_dir_path.mkdir(parents=True)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    utils.set_log_level(args.log_level)

    utils.log("## START ##", "INFO")
    utils.log("Parsed arguments:", "INFO")
    for arg_name, arg_value in vars(args).items():
        utils.log(f"  {arg_name}: {arg_value}", "INFO")

    experiment_name = utils.get_basename(args)
    utils.log(f"Experiment name: {experiment_name}", "INFO")

    if args.dry_run:
        results_output = open(os.devnull, "wt")
        stats_output = open(os.devnull, "wt")
    else:
        results_output = output_writer(
            path=str(args.output_dir_path / (experiment_name + ".test_results.csv")),
            compression=args.output_compression,
        )
        stats_output = output_writer(
            path=str(args.output_dir_path / (experiment_name + ".train_results.csv")),
            compression=args.output_compression,
        )

    utils.set_seed(args.seed)
    utils.log(f"> Seed set: {args.seed}", "INFO")

    device = utils.set_device(args.device)
    utils.log(f"> Device: {device}", "INFO")

    args.func(experiment_name, results_output, stats_output, args, device)

    results_output.close()
    stats_output.close()

    utils.log("## END ##", "INFO")


if __name__ == "__main__":
    main()
