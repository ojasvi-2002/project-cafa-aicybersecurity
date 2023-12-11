from os import path
import torch

from src.models.mlp import LitMLP


def load_trained_model(saved_model_name, models_dir='trained-models/',
                          model_type='mlp', dataset=None):
    """
    Loads the required model from the saved models directory
    """
    saved_model_path = path.join(models_dir, f'{saved_model_name}')
    print(f"Loading model from {saved_model_path}")

    if model_type == 'mlp':
        map_location = None if torch.cuda.is_available() else 'cpu'
        model = LitMLP.load_from_checkpoint(saved_model_path, map_location=map_location)
    elif model_type == 'tabnet':
        # TODO support tabnet
        model_wrapper = TabNetModelWrapper.load_model(model_ref_zip=saved_model_path,
                                       cat_idxs=dataset.unordered_indices, cat_dims=dataset.unordered_dims)
        model = model_wrapper.get_model_object()
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    return model

