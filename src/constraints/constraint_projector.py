import numpy as np

from src.constraints.utilizing.constrainer import Constrainer
import logging

logger = logging.getLogger(__name__)


class ConstraintProjector:
    """
    Class for generically projecting samples onto a given constraint set, thus making them 'feasible'.
        - The projection can be done relative to any constrainer (which implements the Constrainer API).
        - The projection is done by a binary search, which minimizes the amount of freed literals.
    """

    def __init__(self,
                 constrainer: Constrainer,
                 upper_projection_budget_bound: float = 0.5):
        """
        :param constrainer: Object representing the constraints projected onto, implemented with Constrainer API.
        :param upper_projection_budget_bound: The upper bound on the projection budget, i.e., the fraction features
                                              freed in projection.
        """
        self.constrainer: Constrainer = constrainer
        self.upper_projection_budget_bound = upper_projection_budget_bound

    def project(self, sample: np.ndarray, sample_original: np.ndarray) -> (bool, np.ndarray):
        """
        Projects a given sample onto the constraints, while minimizing the amount of freed literals
            by utilizing binary search.
        """

        # Set the projection-budget bounds of the binary search
        lower_phi, upper_phi = 0, round(self.upper_projection_budget_bound * len(sample))

        # Check lower and upper phi
        lower_phi_sat, lower_projected_sample = self._single_project(sample,
                                                                     sample_original=sample_original,
                                                                     n_free_literals=lower_phi)
        upper_phi_sat, upper_projected_sample = self._single_project(sample,
                                                                     sample_original=sample_original,
                                                                     n_free_literals=upper_phi)
        mid_phi, mid_phi_sat, mid_projected_sample = -1, False, None

        # Run binary-search as long as upper-phi is satisfiable and lower-phi < upper-phi (binary search still runs)
        while (not lower_phi_sat) and upper_phi_sat and (lower_phi + 1 < upper_phi):
            # Since we assume lower_phi yield unsat, it must be: XVV or XXV
            mid_phi = (lower_phi + upper_phi) // 2
            assert mid_phi != lower_phi and upper_phi != mid_phi
            mid_phi_sat, mid_projected_sample = self._single_project(sample,
                                                                     sample_original=sample_original,
                                                                     n_free_literals=mid_phi)
            if mid_phi_sat:  # means XVV (lower interval has a potential)
                # we move to search in [low, mid]
                upper_phi, upper_phi_sat, upper_projected_sample = mid_phi, mid_phi_sat, mid_projected_sample
            else:  # means XXV (upper interval has a potential)
                # we move to search in [mid, high]
                lower_phi, lower_phi_sat, lower_projected_sample = mid_phi, mid_phi_sat, mid_projected_sample
        # get the satisfying projected sample with the smallest phi
        if lower_phi_sat:
            # if `lower` is sat, we simply take it
            n_phi, is_sat, projected_sample = lower_phi, lower_phi_sat, lower_projected_sample
        elif mid_phi_sat:
            n_phi, is_sat, projected_sample = mid_phi, mid_phi_sat, mid_projected_sample
        else:
            n_phi, is_sat, projected_sample = upper_phi, upper_phi_sat, upper_projected_sample

        logger.info(f"Projection was {'successful' if is_sat else 'failed'} with budget={n_phi}")

        return is_sat, projected_sample

    def _single_project(self, sample: np.ndarray,
                        sample_original: np.ndarray,
                        n_free_literals: int) -> (bool, np.ndarray):
        """
        Projects a single sample onto the constraints, by freeing `n_free_literals` literals.
            - Defines the general projection scheme, utilizing the Constrainer API.
            - Simple projection, that is employed in `self.project` as part of a bigger binary-search projection scheme.
        """
        sample = sample.copy()

        before_projection_sat = self.constrainer.check_sat(sample, sample_original=sample_original)
        if before_projection_sat or n_free_literals == 0:
            # in case sample obeys the constraints / no projection desired, simply return it
            return before_projection_sat, sample

        # otherwise - project by sat solver:
        # 1. Find the literals to free
        literals_scores = self.constrainer.get_literals_scores(sample)
        literals_to_free = literals_scores.argsort()[:n_free_literals]  # default literals to free
        if len(np.unique(literals_scores)) == 1:
            # Sample from the top `n_free_literals` literals to free, in case of tie
            literals_to_free = np.random.choice(len(literals_scores), size=n_free_literals, replace=False)
        else:
            # TODO make sure the following line is printed occasionally (otherwise, inspect the score system)
            print("sometimes the literals cost differ!", literals_scores)

        # TODO [currently disabled] consider a softmax option (with `p=softmax(-literals_scores))`)
        # temp_factor = 1  # the lower - the more uniform (0 -> uniform)
        # softmax = lambda x: np.exp(temp_factor * x) / np.sum(np.exp(temp_factor * x))

        # 2. Attempt to project the sample
        is_projection_succ, projected_sample = self.constrainer.project_sample(sample, literals_to_free,
                                                                               sample_original=sample_original)

        return is_projection_succ, projected_sample
