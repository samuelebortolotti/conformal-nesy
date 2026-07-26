"""Module used for global utils functions"""

import logging
import random
import torch

import numpy as np

LOG_LEVEL = logging.INFO


def set_log_level(level_str):
    global LOG_LEVEL
    LOG_LEVEL = getattr(logging, level_str.upper(), logging.INFO)


def log(message, level_str="INFO"):
    """
    Logs a message if the message's level is >= current global LOG_LEVEL.
    """
    numeric_level = getattr(logging, level_str.upper(), logging.INFO)
    if numeric_level >= LOG_LEVEL:
        logging.log(numeric_level, message)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_basename(args, separator="_"):
    # Keys never included in the experiment name
    always_exclude = {"func", "model_path", "model_dir", "alpha_concepts", "alpha_label", "cal_ratio"}

    # epsilon_symbols / epsilon_rules were added to the global argparser for DSL support,
    # but DPL/LTN/CBM checkpoints were saved before these args existed.  Exclude them
    # from the basename for any model that is not DSL or linear_predictor so that
    # checkpoint lookups for those older models still resolve correctly.
    nesy = getattr(args, "nesy", None)
    if nesy not in ("dsl", "linear_predictor"):
        always_exclude |= {"epsilon_symbols", "epsilon_rules"}

    args_str = []
    for key, value in vars(args).items():
        if key not in always_exclude:
            args_str.append(str(value).replace("/", "-"))
    experiment_name = separator.join(args_str)
    return f"experiment_{experiment_name}"


def set_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
