from typing import Union, Optional, List
import logging

import numpy as np
from tqdm import tqdm

from art.attacks import EvasionAttack
from art.estimators import BaseEstimator, LossGradientsMixin
from art.estimators.classification import ClassifierMixin
from art.summary_writer import SummaryWriter

logger = logging.getLogger(__name__)

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
            ordinal_indices: np.ndarray,
            cont_indices: np.ndarray,
            feature_ranges: np.ndarray,  # shape (n_features, 2) with min/max values for each feature

            cat_encoding_method: str = 'one_hot_encoding',
            one_hot_groups: List[np.ndarray] = None,

            # TabPGD HPs:
            random_init: bool = True,  # TODO add seed to reprod
            # batch_size: int = None,  # TODO add support for batching
            eps: float = 0.03,
            step_size: float = 0.0003,
            max_iter: int = 100,
            perturb_categorical_each_steps: int = 10,

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
        self.eps = eps
        self.step_size = step_size
        self.max_iter = max_iter
        self.perturb_categorical_each_steps = perturb_categorical_each_steps

        self.feature_ranges = feature_ranges
        self.standard_factors = standard_factors
        self.cat_indices = cat_indices
        self.ordinal_indices = ordinal_indices
        self.cont_indices = cont_indices

        self.cat_encoding_method = cat_encoding_method
        self.one_hot_groups = one_hot_groups

        # validations:
        self._validate_input()

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
                     Note that a mask should apply to One-Hot groups altogether.
        :return: An array holding the adversarial examples.
        """

        x, y = x.astype(np.float32), y.astype(np.int64)
        allow_updates = np.ones(x.shape[0], dtype=np.float32)
        accum_grads = np.zeros_like(x)
        mask = np.ones_like(x) if mask is None else mask
        mask = mask.astype(np.float32)

        # TODO epsilon seem to be the most bottle neck for ordinal features
        # INSPECTED = [2]  # debugging line
        epsilon_ball_upper = x + (self.eps * self.standard_factors)
        epsilon_ball_upper[:, self.ordinal_indices] = np.ceil(epsilon_ball_upper[:, self.ordinal_indices])
        # epsilon_ball_upper[:, INSPECTED] = np.ceil(epsilon_ball_upper[:, INSPECTED]) +10_000
        epsilon_ball_upper[:, self.cat_indices] = 1.0  # TODO generalize (this is for one-hot)

        epsilon_ball_lower = x - (self.eps * self.standard_factors)
        epsilon_ball_lower[:, self.ordinal_indices] = np.floor(epsilon_ball_lower[:, self.ordinal_indices])
        # epsilon_ball_lower[:, INSPECTED] = np.floor(epsilon_ball_lower[:, INSPECTED]) -10_000
        epsilon_ball_lower[:, self.cat_indices] = 0.0  # TODO generalize
        # 0. preliminary summary writer

        # 0. Init random perturbation in epsilon-ball # TODO
        x_adv = x.copy()

        for step in tqdm(range(self.max_iter)):
            x_adv_before_perturb = x_adv.copy()  # todo this is for validation only

            # Inject allow_updates-mask to 'mask'
            mask *= allow_updates[:, None]
            perturbation_temp = np.zeros_like(x_adv)

            if step == 1:  # TODO hack make more elegant
                accum_grads = np.zeros_like(x_adv)

            if self.random_init and step == 0:
                # if random_init is True, then we start the attack from a random point inside the epsilon-ball
                perturbation_temp = np.random.uniform(-self.eps, self.eps, x.shape).astype(
                    np.float32) * self.standard_factors
                accum_grads = perturbation_temp
            else:
                # Get gradient wrt loss; invert it if attack is targeted
                # TODO: should the mask be applied before calculating the loss or something?
                grad = self.estimator.loss_gradient(x, y)  # * (1 - 2 * int(self.targeted)) TODO targeted
                # Update accumulated gradients
                accum_grads += grad * mask  # TODO required?
                # Set the temporal perturbation (each feature alters it according to its character)
                perturbation_temp = self.step_size * self.standard_factors * np.sign(grad)

            # 4.1. Perturb continuous as is
            x_adv += self._get_perturbation_continuous(perturbation_temp) * mask
            # 4.2. Perturb integers with rounding
            x_adv += self._get_perturbation_ordinal(perturbation_temp) * mask
            # 4.3. Perturb categorical according to accumulated grads
            if step % self.perturb_categorical_each_steps == 0 and step > 0:
                x_adv += self._get_perturbation_categorical(x_adv, accum_grads) * mask

            # 5. Project back to standard-epsilon-ball
            x_adv = np.clip(x_adv, epsilon_ball_lower, epsilon_ball_upper)

            # 6.1. Clip to integer features
            x_adv[:, self.ordinal_indices] = np.round(x_adv[:, self.ordinal_indices])
            # 6.2. Clip to feature ranges
            x_adv = np.clip(x_adv, self.feature_ranges[:, 0], self.feature_ranges[:, 1])

            # Assert that non masked / early-stopped feature was perturbed (x_adv_before_perturb)
            assert np.allclose(x_adv[~mask.astype(bool)], x_adv_before_perturb[~mask.astype(bool)], atol=1e-6)
            assert np.allclose(x_adv[~allow_updates.astype(bool), :],
                               x_adv_before_perturb[~allow_updates.astype(bool), :], atol=1e-6)

            # 8. Early stop who are already adversarial
            x_adv = x_adv.astype(np.float32)
            is_attack_success = self.estimator.predict(x_adv).argmax(axis=1) != y
            logger.info(f"ASR: {is_attack_success.mean() * 100: .2f}%")
            allow_updates -= allow_updates * is_attack_success

            # 9. Track attack metrics # TODO

            # 10. Early stop if all samples are already adversarial
            if allow_updates.sum() == 0:
                break

        return x_adv

    def _get_perturbation_continuous(self, perturbation_temp):
        perturb_cont = np.zeros_like(perturbation_temp)
        perturb_cont[:, self.cont_indices] = perturbation_temp[:, self.cont_indices]
        return perturb_cont

    def _get_perturbation_ordinal(self, perturbation_temp):
        perturb_ord = np.zeros_like(perturbation_temp)
        perturb_ord[:, self.ordinal_indices] = (np.ceil(np.abs(perturbation_temp[:, self.ordinal_indices]))
                                                * np.sign(perturbation_temp[:, self.ordinal_indices]))
        return perturb_ord

    def _get_perturbation_categorical(self, x_adv, accum_grads,
                                      perturb_one_feature_only=False):
        perturb_cat = np.zeros_like(x_adv)

        if self.cat_encoding_method == 'one_hot_encoding':  # TODO generalize to TabNet

            # get the max value of each OH group
            score_grads_per_group = np.zeros((x_adv.shape[0], len(self.one_hot_groups)))
            for id_oh_group, oh_group in enumerate(self.one_hot_groups):
                score_grads_per_group[:, id_oh_group] = accum_grads[:, oh_group].max(-1)

            # perturb groups with the largest max value
            for id_oh_group, oh_group in enumerate(self.one_hot_groups):
                samples_to_update_indices = np.arange(x_adv.shape[0])
                if perturb_one_feature_only:
                    # get indices of samples we want to update (samples with max grad in this group)
                    samples_to_update = (score_grads_per_group.argmax(axis=-1) == id_oh_group)
                    samples_to_update_indices = np.where(samples_to_update)[0]
                samples_to_update_indices = np.expand_dims(samples_to_update_indices, axis=1)  # to be used as an index
                # get the largest accum_grads of the group
                chosen_cats = oh_group[accum_grads[:, oh_group].argmax(axis=1)]
                chosen_cats = chosen_cats[samples_to_update_indices]
                # turn (only) these to 1 after cancelling previous category
                perturb_cat[samples_to_update_indices, oh_group] = -x_adv[samples_to_update_indices, oh_group]
                perturb_cat[samples_to_update_indices, chosen_cats] += 1
        else:
            raise NotImplementedError  # TODO
        return perturb_cat

    def _validate_input(self):
        assert self.cat_encoding_method in ['one_hot_encoding'], 'only one-hot encoding is supported for now'

        # verify one-hot groups are cover the categorical features
        oh_indices = set()
        for oh_group in self.one_hot_groups:
            oh_indices.update(oh_group)
        assert oh_indices == set(self.cat_indices), 'one-hot groups should cover all categorical indices'

        assert (set(self.cat_indices) | set(self.ordinal_indices) | set(self.cont_indices)
                == set(range(self.feature_ranges.shape[0]))), 'indices should form all features'

        # verify feature indices are disjoint
        assert len(set(self.cat_indices) & set(self.ordinal_indices)) == 0, \
            'cat and ordinal indices should be disjoint'
        assert len(set(self.cat_indices) & set(self.cont_indices)) == 0, \
            'cat and cont indices should be disjoint'
        assert len(set(self.cont_indices) & set(self.ordinal_indices)) == 0, \
            'cont and ordinal indices should be disjoint'
