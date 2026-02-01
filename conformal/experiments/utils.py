import numpy as np
import torch
import copy

from sklearn.metrics import f1_score
from conformal.models.dpl import DPL
from conformal.utils.factories import NetworkFactory


def collect_predictions(model, data_loader, device, multiclass=False, multilabel=False):
    """
    Compute loss and F1 score for the dataset (train or validation).
    Returns raw logits/probabilities after None.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_g = []
    all_conc_pred = []
    all_output_raw = []

    for data, concepts, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output, conc_pred, _ = model(data, eval=True)

        all_preds.append(output.argmax(dim=-1).cpu().numpy())
        all_labels.append(target.cpu().numpy())

        all_conc_pred.append(conc_pred.detach().cpu().numpy())
        all_g.append(concepts.cpu().numpy())
        all_output_raw.append(output.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_conc_pred = np.concatenate(all_conc_pred)
    all_g = np.concatenate(all_g)
    all_output_raw = np.concatenate(all_output_raw)

    if all_conc_pred.ndim == 2:
        all_c = all_conc_pred.argmax(axis=1)
    elif all_conc_pred.ndim == 3:
        all_c = all_conc_pred.argmax(axis=2)
    else:
        all_c = all_conc_pred.argmax(axis=3).squeeze(1)

    return (
        all_labels,
        all_preds,
        all_g.flatten() if not multiclass else all_g,
        all_c.flatten() if not multiclass else all_c,
        None,
        all_output_raw,
        all_conc_pred,
    )


def load_single_cbm(
    n_images, output_dim, input_dim, concept_dim, args, device, initial_state=None
):
    base_model = NetworkFactory.get_network(args.model, input_dim, concept_dim, args)
    model = DPL(
        n_images,
        base_model,
        args.entangled,
        concept_dim,
        output_dim,
        args.dataset,
        True,
    )
    model.to(device)

    if initial_state is not None:
        model.load_state_dict(initial_state)
    else:
        initial_state = copy.deepcopy(model.state_dict())

    return model, initial_state


def load_model(model, model_path, device):
    if model_path.exists():
        loaded_content = torch.load(str(model_path), map_location=device)

        if isinstance(loaded_content, dict):
            model.load_state_dict(loaded_content)
        else:
            model = loaded_content
    else:
        raise FileNotFoundError(f"No model file found at: {model_path}")
    return model
