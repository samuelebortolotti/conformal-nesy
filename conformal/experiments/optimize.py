import optuna
import tempfile
import traceback
import copy
import torch
import json
from datetime import datetime
from pathlib import Path

from conformal.experiments.train import main as train_main
from conformal.experiments.train import train_parser


def configure_subparsers(subparsers):
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
    parser.add_argument(
        "--storage-url", type=str, default=None,
        help="Optuna storage URL for parallel workers (e.g. sqlite:///optuna.db)"
    )
    parser.add_argument(
        "--study-name", type=str, default=None,
        help="Optuna study name (required when using --storage-url)"
    )
    parser.set_defaults(func=main)


def objective(trial, base_args, experiment_name, output_dir, device, failure_log):
    args = copy.deepcopy(base_args)

    args.learning_rate = trial.suggest_categorical(
        "learning_rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    )
    args.momentum = trial.suggest_categorical(
        "momentum", [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99]
    )
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
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
    elif args.nesy == "cbm":
        pass  # no additional hyperparameters

    args.model_path = f"trial_{trial.number}.pt"
    args.output_dir_path = output_dir
    args.dry_run = True

    def pruning_callback(epoch, val_f1):
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False
    ) as stats_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=False
    ) as results_file:

        try:
            return train_main(
                experiment_name=f"trial_{trial.number}",
                results_output_h=results_file,
                stats_output_h=stats_file,
                args=args,
                device=device,
                pruning_callback=pruning_callback,
            )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(failure_log, "a") as fh:
                fh.write(f"[{ts}] Trial {trial.number} FAILED\n")
                fh.write(f"  params: {trial.params}\n")
                fh.write(f"  error:  {type(exc).__name__}: {exc}\n")
                fh.write(traceback.format_exc())
                fh.write("\n")
            raise
        finally:
            torch.cuda.empty_cache()


def run_study(
    device,
    args,
    experiment_name,
    dataset,
    storage=None,
    study_name="cbm_optuna",
    direction="maximize",
):
    output_dir = Path("./" + args.optuna_path)
    output_dir.mkdir(exist_ok=True)
    device = device
    failure_log = output_dir / "failed_trials.log"

    # create sampler
    sampler = optuna.integration.BoTorchSampler(
        n_startup_trials=5,
        independent_sampler=optuna.samplers.TPESampler(seed=args.seed),
        seed=args.seed,
    )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    if storage:
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    study.optimize(
        lambda trial: objective(
            trial=trial,
            base_args=args,
            experiment_name=experiment_name,
            output_dir=output_dir,
            device=device,
            failure_log=failure_log,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Trials completed: {len(completed)}, pruned: {len(pruned)}, failed: {len(failed)}")
    if failed:
        print(f"  Failure details written to: {failure_log}")
    if completed:
        print("Best trial:")
        print("  Value (F1):", study.best_trial.value)
        print("  Params:", study.best_trial.params)
    else:
        print("No trials completed successfully. Check the failure log for details.")

    results = {
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "n_failed": len(failed),
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "state": str(t.state),
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    if completed:
        results["best_trial"] = {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        }

    results_file = output_dir / f"results_{dataset}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

    return study


def main(experiment_name, results_output_h, stats_output_h, args, device):
    storage   = getattr(args, "storage_url",  None)
    study_name = getattr(args, "study_name", None) or f"ltn_{args.dataset}_cs{args.concept_supervision}"
    run_study(
        device=device,
        args=args,
        experiment_name=experiment_name,
        dataset=args.dataset,
        storage=storage,
        study_name=study_name,
    )
