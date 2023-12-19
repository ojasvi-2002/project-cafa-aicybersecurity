import logging
from typing import List, Dict

from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import numpy as np
from art.estimators.classification import PyTorchClassifier

from src.attacks.cafa import CaFA
from src.constraints.constraint_projector import ConstraintProjector
from src.constraints.utilizing.constrainer import DCsConstrainer
from src.models.utils import load_trained_model
from src.utils import evaluate_crafted_samples
from src.datasets.load_tabular_data import TabularDataset
from src.constraints.mining.mine_dcs import mine_dcs
from src.models.mlp import grid_search_hyperparameters, train

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Used config: {OmegaConf.to_yaml(cfg)}")

    # 1. Process data:
    tab_dataset = TabularDataset(**cfg.data.params)

    # 2. Load model; optionally, re-train before:
    if cfg.ml_model.perform_training or cfg.ml_model.perform_grid_search_hparams:
        best_hparams = cfg.ml_model.default_hparams
        if cfg.ml_model.perform_grid_search_hparams:
            best_hparams = grid_search_hyperparameters(trainset=tab_dataset.trainset,
                                                       testset=tab_dataset.testset,
                                                       tab_dataset=tab_dataset)
        train(best_hparams, trainset=tab_dataset.trainset, testset=tab_dataset.testset, tab_dataset=tab_dataset,
              model_artifact_path=cfg.ml_model.model_artifact_path)
    model = load_trained_model(cfg.ml_model.model_artifact_path, model_type=cfg.ml_model.model_type)

    # 3. Wrap the model to ART classifier, for executing the attack:
    def model_loss(output, target):  # TODO resolve this hack
        output = output.float()
        target = target.long()
        return torch.functional.F.cross_entropy(output, target)

    classifier = PyTorchClassifier(
        model=model,
        loss=model_loss,
        input_shape=tab_dataset.n_features,
        nb_classes=tab_dataset.n_classes,
    )
    eval_params = dict(classifier=classifier, tab_dataset=tab_dataset)

    # 4. Load constraints; Optionally, mine them before:
    if 'constraints' in cfg and cfg.constraints:
        mining_source_params = cfg.data.params.copy()
        mining_source_params['encoding_method'] = None  # we set default (label-) encoding for constraint mining
        tab_dcs_dataset = TabularDataset(**mining_source_params)

        # [Optionally] Mine the DCs:
        mine_dcs(
            x_mine_source_df=tab_dcs_dataset.x_df,
            x_dcs_col_names=tab_dcs_dataset.x_dcs_col_names,
            **cfg.constraints.mining_params
        )

        # Initialize the DCs constrainer:
        constrainer = DCsConstrainer(
            x_tuples_df=tab_dcs_dataset.x_df,
            **tab_dcs_dataset.structure_constraints,
            **cfg.constraints.constrainer_params
        )

        # Initialize the generic constraints projector
        projector = ConstraintProjector(
            constrainer=constrainer,
            **cfg.constraints.projector_params
        )

        eval_params.update(dict(constrainer=constrainer, tab_dataset_constrainer=tab_dcs_dataset))

    # 5. Evaluate before the attack:
    X, y = tab_dataset.X_test[:cfg.n_samples_to_attack], tab_dataset.y_test[:cfg.n_samples_to_attack]
    evaluations: Dict[str, Dict[str, float]] = {}

    evaluations['before-attack'] = evaluate_crafted_samples(X_adv=X, X_orig=X, y=y, **eval_params)
    logger.info(f"before-attack: {evaluations['before-attack']}")

    # 4. Attack:
    X_adv = None
    if cfg.perform_attack:
        attack = CaFA(estimator=classifier,
                      **tab_dataset.structure_constraints,
                      **cfg.attack)
        X_adv = attack.generate(x=X, y=y)

        evaluations['after-cafa'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
        logger.info(f"after-cafa: {evaluations['after-cafa']}")

    # 5. Project
    if 'constraints' in cfg and cfg.constraints and cfg.perform_projection and X_adv is not None:
        # collect sample projected to numpy array
        X_adv_proj = []

        for x_orig, x_adv in zip(X, X_adv):  # for validation

            # 5.A. Transform sample to the format of the DCs dataset
            sample_orig = TabularDataset.cast_sample_format(x_orig, from_dataset=tab_dataset,
                                                            to_dataset=tab_dcs_dataset)
            sample_adv = TabularDataset.cast_sample_format(x_adv, from_dataset=tab_dataset, to_dataset=tab_dcs_dataset)

            # 5.A.1. Sanity checks:
            assert np.all(x_orig ==
                          TabularDataset.cast_sample_format(sample_orig, from_dataset=tab_dcs_dataset,
                                                            to_dataset=tab_dataset))
            assert np.all(sample_orig ==
                          TabularDataset.cast_sample_format(x_orig, from_dataset=tab_dataset,
                                                            to_dataset=tab_dcs_dataset))

            # 5.B. Project
            is_succ, sample_projected = projector.project(sample_adv, sample_original=sample_orig)

            # 5.C. Transform back to the format of the model input
            x_adv_proj = TabularDataset.cast_sample_format(sample_projected, from_dataset=tab_dcs_dataset,
                                                           to_dataset=tab_dataset)
            X_adv_proj.append(x_adv_proj)

        X_adv_proj = np.array(X_adv_proj)

        evaluations['after-cafa-projection'] = evaluate_crafted_samples(X_adv=X_adv_proj, X_orig=X, y=y, **eval_params)
        logger.info(f"after-projection: {evaluations['after-cafa-projection']}")

    logger.info("Finished attack.")
    logger.info(evaluations)


if __name__ == "__main__":
    main()
