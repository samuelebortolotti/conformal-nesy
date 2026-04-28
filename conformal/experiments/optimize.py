import optuna
import tempfile
import copy
from pathlib import Path

from conformal.experiments.train import main as train_main
from conformal.experiments.train import train_parser


def configure_subparsers(subparsers):
    """Configure the subparsers."""
    parser = subparsers.add_parser(
        "optuna",
        help="Trains a nn model on a dataset",
    )
    train_parser(parser)
    parser.add_argument(
        "--optuna-path", type=str, default="optuna_runs", help="Optuna path"
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Trials")
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout in seconds per study (None = no limit)"
    )
    parser.set_defaults(func=main)


def objective(trial, base_args, experiment_name, output_dir, device):
    args = copy.deepcopy(base_args)

    args.learning_rate = trial.suggest_categorical(
        "learning_rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    )
    args.momentum = trial.suggest_categorical(
        "momentum", [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99]
    )
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    args.opt = trial.suggest_categorical("opt", ["adam", "sgd"])

    if args.nesy == "ltn":
        # consistent logic family

        logic_family = trial.suggest_categorical(
            "logic_family", ["godel", "product", "lukasiewicz"]
        )

        if logic_family == "godel":
            args.and_op = "godel"
            args.or_op = "godel"
            args.imp_op = "godel"

        elif logic_family == "product":
            args.and_op = "prod"
            args.or_op = "prod"
            args.imp_op = "goguen"

        elif logic_family == "lukasiewicz":
            args.and_op = "luk"
            args.or_op = "luk"
            args.imp_op = "luk"

        args.p = trial.suggest_categorical("p", list(range(1, 10)))
    elif args.nesy == "dsl":
        args.epsilon_symbols = trial.suggest_float(
            "epsilon_symbols", 1e-5, 0.5, log=True
        )
        args.epsilon_rules = trial.suggest_float("epsilon_rules", 1e-5, 0.5, log=True)

    args.model_path = f"trial_{trial.number}.pt"
    args.output_dir_path = output_dir
    args.dry_run = True

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False
    ) as stats_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=False
    ) as results_file:

        return train_main(
            experiment_name=f"trial_{trial.number}",
            results_output_h=results_file,
            stats_output_h=stats_file,
            args=args,
            device=device,
        )


def run_study(
    device,
    args,
    experiment_name,
    storage=None,
    study_name="cbm_optuna",
    direction="maximize",
):
    output_dir = Path("./" + args.optuna_path)
    output_dir.mkdir(exist_ok=True)
    device = device

    # create sampler
    sampler = optuna.integration.BoTorchSampler(
        n_startup_trials=10,
        independent_sampler=optuna.samplers.TPESampler(seed=args.seed),
        seed=args.seed,
    )

    if storage:
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction=direction, sampler=sampler)

    study.optimize(
        lambda trial: objective(
            trial=trial,
            base_args=args,
            experiment_name=experiment_name,
            output_dir=output_dir,
            device=device,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print("Best trial:")
    print("  Value (F1):", study.best_trial.value)
    print("  Params:", study.best_trial.params)
    return study


def main(experiment_name, results_output_h, stats_output_h, args, device):
    run_study(device=device, args=args, experiment_name=experiment_name)
