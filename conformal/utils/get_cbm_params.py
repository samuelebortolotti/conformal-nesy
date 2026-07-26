"""Read per-dataset CBM best hyperparameters from an Optuna results file.

Usage: python conformal/utils/get_cbm_params.py <dataset> <optuna_path>
Output: --learning-rate X --momentum X --batch-size X --opt X
"""
import json
import sys
import pathlib

dataset = sys.argv[1]
optuna_path = pathlib.Path(sys.argv[2])
results = json.load(open(optuna_path / f"results_{dataset}.json"))
p = results["best_trial"]["params"]
print(
    f"--learning-rate {p['learning_rate']} "
    f"--momentum {p['momentum']} "
    f"--batch-size {p['batch_size']} "
    f"--opt {p['opt']}"
)
