from typing import Union, Optional, List
import numpy as np
from tqdm import tqdm

from art.attacks import EvasionAttack
from art.estimators import BaseEstimator, LossGradientsMixin
from art.estimators.classification import ClassifierMixin
from art.summary_writer import SummaryWriter


class TabPGD(EvasionAttack):
    """
    PGD attack on tabular data.
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
        "summary_writer",
    ]
    # Requiring implementation of 'loss_gradient()' (i.e., white-box access), via `LossGradientsMixin`.
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(
            self,
            estimator: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",

            # Data-specific parameters:
            standard_factors: np.ndarray,
            cat_indices: np.ndarray,
            ordinal_indices: np.ndarray,  # disjoint from cat_indices (todo validate);
            cont_indices: np.ndarray,  # disjoint from rest, complementay (todo validate);
            feature_ranges: np.ndarray,  # shape (n_features, 2) with min/max values for each feature

            cat_encoding_method: str = 'one_hot_encoding',  # TODO more
            one_hot_groups: List[np.ndarray] = None,

            # TabPGD HPs:
            random_init: bool = True,
            batch_size: int = 3,
            eps: float = 0.03,
            step_size: float = 0.01,
            max_iter: int = 100,

            # Misc:  # TODO integrate summary_writer in the code
            summary_writer: Union[str, bool, SummaryWriter] = False,
    ):
        """
        Create a `TabPGD` attack instance.

        :param estimator: A trained classifier.
        :param summary_writer: Activate summary writer for TensorBoard.
        """
        super().__init__(estimator=estimator,
                         summary_writer=summary_writer)
        self.random_init = random_init
        self.batch_size = batch_size
        self.eps = eps
        self.step_size = step_size
        self.max_iter = max_iter

        self.feature_ranges = feature_ranges
        self.standard_factors = standard_factors
        self.cat_indices = cat_indices
        self.ordinal_indices = ordinal_indices
        self.cont_indices = cont_indices

        self.cat_encoding_method = cat_encoding_method
        self.one_hot_groups = one_hot_groups

    def generate(self,
                 x: np.ndarray,
                 y: np.ndarray = None,
                 mask: np.ndarray = None,
                 **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: An array with the original labels to be predicted.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
        :return: An array holding the adversarial examples.
        """
        # TODO batch it

        x, y = x.astype(np.float32), y.astype(np.int64)
        allow_updates = np.ones(x.shape[0], dtype=np.float32)
        accum_grads = np.zeros_like(x)
        mask = np.ones_like(x, dtype=bool) if mask is None else mask

        epsilon_ball_upper = x + (self.eps * self.standard_factors)
        epsilon_ball_lower = x - (self.eps * self.standard_factors)
        # 0. preliminary summary writer

        # 0. Init random perturbation in epsilon-ball # TODO
        x_adv = x.copy()

        for step in tqdm(range(self.max_iter)):
            x_adv_before_perturb = x_adv.copy()  # todo this is for validation only

            # Get gradient wrt loss; invert it if attack is targeted
            # TODO should the mask be applied before calculating the loss or something?
            grad = self.estimator.loss_gradient(x, y)  #* (1 - 2 * int(self.targeted)) TODO targeted
            # Update accumulated gradients
            accum_grads += grad

            # Mask non-perturbed features
            grad = grad * mask
            # Mask early-stopped samples
            grad = grad * allow_updates[:, None]

            # Set the temporal perturbation (each feature alters it according to its character)
            perturbation_temp = self.step_size * self.standard_factors * np.sign(grad)

            # 4.1. Perturb continuous as is
            x_adv[:, self.cont_indices] += perturbation_temp[:, self.cont_indices]
            # 4.2. Perturb integers with rounding
            x_adv[:, self.ordinal_indices] = np.round(x_adv[:, self.ordinal_indices] +
                                                      perturbation_temp[:, self.ordinal_indices])
            # 4.3. Perturb categorical according to accumulated grads
            # TODO make 'masking' and 'allow_update' work here
            if self.cat_encoding_method == 'one_hot_encoding':  # TODO generalize to TabNet
                for oh_group in self.one_hot_groups:
                    # get the largest accum_grads of the group
                    chosen_cat_idx = oh_group[accum_grads[:, oh_group].argmax(axis=1)]
                    # turn (only) these to 1
                    x_adv[:, oh_group] = 0
                    x_adv[np.arange(x_adv.shape[0]), chosen_cat_idx] = 1
                    # TODO can also implement it ONLY on the largest gradient group
            else:
                raise NotImplementedError  # TODO
            # 5. Project back to standard-epsilon-ball
            x_adv = np.clip(x_adv, epsilon_ball_lower, epsilon_ball_upper)
            # 6. Clip to feature ranges
            x_adv = np.clip(x_adv, self.feature_ranges[:, 0], self.feature_ranges[:, 1])

            # assert that non masked / early-stopped feature was perturbed (x_adv_before_perturb)
            assert np.all(x_adv[~mask] == x_adv_before_perturb[~mask])
            assert np.all(x_adv[~allow_updates.astype(bool), :] == x_adv_before_perturb[~allow_updates.astype(bool), :])

            # 8. Early stop who are already adversarial
            x_adv = x_adv.astype(np.float32)
            is_attack_success = self.estimator.predict(x_adv).argmax(axis=1) != y
            allow_updates -= allow_updates * is_attack_success

            # 9. Summary writer # TODO

            # 10. Early stop if all samples are already adversarial
            if allow_updates.sum() == 0:
                break

        return x_adv
