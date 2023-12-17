import json
import os
import sys
from typing import List
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

from src.constraints.modeling.dcs_model import DenialConstraint

PATH_TO_DC_MINER_JAR = "resources/DCFinder_ADC-1.0-SNAPSHOT.jar"  # path is relative to this pacakge


def mine_dcs(x_mine_source_df: pd.DataFrame,
             raw_dcs_out_path: str,
             evaluated_dcs_out_path: str,

             # DCs configuration
             approx_violation_threshold: float = 0.01,
             n_tuples_to_eval: int = 750,  # limit the recorded best-other-tuples to 0.75K
             n_dcs_to_eval: int = 10_000,  # limit the number of evaluated DCs to 10K

             # Phases to execute:
             perform_constraints_mining: bool = True,
             perform_constraints_eval: bool = True,
             ):
    if perform_constraints_mining:
        print(">> Running DC Mining Algorithm")
        run_fast_adc(mine_source_df=x_mine_source_df,
                     path_to_save_raw_dcs=raw_dcs_out_path,
                     approx_violation_threshold=approx_violation_threshold)

    if perform_constraints_eval:
        print(">> Evaluating and Ranking DCs")
        dcs: List[DenialConstraint] = load_dcs_from_txt(raw_dcs_out_path)
        # Evaluate DCs metrics and Rank DCs by these metrics (via manually-crafted linear combination)
        evaluated_dcs = eval_and_rank_dcs(
            x_tuples_df=x_mine_source_df,
            dcs=dcs,
            n_tuples_to_eval=n_tuples_to_eval,
            n_dcs_to_eval=n_dcs_to_eval,
        )
        evaluated_dcs.to_csv(evaluated_dcs_out_path, index=False)


# TODO make this mining operational
# TODO what was changed in DCs source code? is it configurable out-of-the-box?
def run_fast_adc(mine_source_df: pd.DataFrame,
                 path_to_save_raw_dcs: str,
                 approx_violation_threshold: float = 0.01):
    # get the script's dir:
    curr_package_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    path_to_jar = os.path.join(curr_package_dir, PATH_TO_DC_MINER_JAR)

    # save mine_source_df to 'input_processed_data_csv_name'
    input_processed_data_csv_name = os.path.join(curr_package_dir, "input_processed_data.csv")
    mine_source_df.to_csv(input_processed_data_csv_name, index=False)

    # run:
    print(">> Running DC Mining Algorithm")
    res = subprocess.run(["java", "-jar", path_to_jar, input_processed_data_csv_name, str(approx_violation_threshold)])
    print(res, res.stdout, res.stderr, sep="\n----")
    # TODO make sure it was saved to `path_to_save_raw_dcs`


def load_dcs_from_txt(dcs_txt_path: str) -> List[DenialConstraint]:
    dcs = []
    # Loads constraints from txt
    with open(dcs_txt_path, "r") as f:
        for dc_file_idx, dc_string in enumerate(f):
            dcs.append(DenialConstraint(dc_string=dc_string,
                                        dc_file_idx=dc_file_idx))
    return dcs


def eval_and_rank_dcs(x_tuples_df: pd.DataFrame,
                      dcs: List[DenialConstraint],
                      n_dcs_to_eval: int = 10_000,
                      n_tuples_to_eval: int = 750):
    """
    :param x_tuples_df: DataFrame with data tuples to evaluate on. These tuples will play the role of main tuple
                        (in the pair).
                        Should be trimmed BEFORE this function.
    :param dcs: list with DC objects to evaluate.
                We assume that all DCs have the same size of `dc.data_tuple` (denoted `n_other_data_tuples`)
    :param n_dcs_to_eval: amount of DC to evaluate (evaluate the `n_dcs` most succinctness DCs)
    :param n_tuples_to_eval: How many of the best"other_tuples" should we keep for each DC.
    :return: prints and saves evaluation CSV.
    """
    # Trim DC as requested, by _Succinctness_ - measure the length of each DC  (the closer it is to 1,
    # the more compact the DC)
    _dc_sizes = [dc.get_predicate_count() for dc in dcs]
    _min_dc_size = min(_dc_sizes)
    succinctness_per_dc = _min_dc_size / np.array(_dc_sizes)
    if n_dcs_to_eval:
        # trim by succinctness
        top_succinct_dcs = succinctness_per_dc.argsort()[-n_dcs_to_eval:]
        old_dcs, dcs = dcs, []
        for old_dc_idx in top_succinct_dcs:
            dcs.append(old_dcs[old_dc_idx])

        # calculate succinctness again
        _dc_sizes = [dc.get_predicate_count() for dc in dcs]
        succinctness_per_dc = _min_dc_size / np.array(_dc_sizes)

    x_tuples_df = x_tuples_df[:n_tuples_to_eval].copy()

    # Set the 'other-tuples' data for each DC
    for dc in dcs:
        dc.set_other_tuples_data(x_tuples_df)

    print(f"Evaluating {len(dcs)=}, `t` from {len(x_tuples_df)=} and `t'` from {len(x_tuples_df)=}")

    # Metric I (g_1): for each DC we calculate the violation rate (over all the possible pairs)
    ##      [Analogue to f_1 from Livshits et al. 2021 / g_1 in FastADC]
    pairs_violation_rate_per_dc = np.zeros(len(dcs))
    # Metric II (g_2): proportion of tuples with _any_ violation of the DC
    tuple_violation_rate_per_dc = np.zeros(len(dcs))
    """ # disabled at the moment
    # Metric III: proportion of tuples that perfectly satisfy the dc
    # # the entry [t_idx, dc_idx] records the violation count the tuple `t_idx`  the DC `dc_idx`.
    tuple_violation_count_per_dc_per_tuple = np.zeros((len(dcs), len(df)))
    """
    # Coverage - rates the amount of predicates-sat in a certain DC
    #       entry [dc_idx, sat_pred_count_idx] = how many pairs satisfied `sat_pred_count_idx` predicates in `dc_idx`
    sat_pred_count_per_dc = np.zeros((len(dcs), max(_dc_sizes) + 1))
    best_other_tuples_per_dc = np.zeros((len(dcs), n_tuples_to_eval), dtype=int)

    for dc_idx, dc in tqdm(enumerate(dcs), desc="Evaluating DCs..."):
        dc_sat_per_other_tuples = np.zeros(n_tuples_to_eval, dtype=int)
        for idx1 in range(len(x_tuples_df)):  # iterate on t (main tuple)
            is_sat_arr, sat_predicates_count_arr = dc.check_satisfaction_all_pairs(x_tuples_df.iloc[idx1].to_dict())

            # Track the sat of tuples playing the "other-tuple" role
            dc_sat_per_other_tuples += is_sat_arr.values

            # Metrics:
            pairs_violation_rate_per_dc[dc_idx] += (~is_sat_arr).sum()  # Metric I
            tuple_violation_rate_per_dc[dc_idx] += (~is_sat_arr).any()  # Metric II
            # Coverage:
            for dc_pred_count_idx, dc_pred_count_val in \
                    zip(*np.unique(sat_predicates_count_arr, return_counts=True)):
                # aggregates the satisfied-predicates spotted in the `dc` with `idx1`.
                sat_pred_count_per_dc[dc_idx, int(dc_pred_count_idx)] += dc_pred_count_val

        best_other_tuples_per_dc[dc_idx] = dc_sat_per_other_tuples.argsort()[-n_tuples_to_eval:][::-1]
        print(dc_idx, ">>", dc_sat_per_other_tuples.min(), dc_sat_per_other_tuples.max())

    # Normalize metrics
    pairs_violation_rate_per_dc /= n_tuples_to_eval * n_tuples_to_eval  # normalize by number of pairs
    tuple_violation_rate_per_dc /= n_tuples_to_eval
    # Calculate coverage
    w = (np.arange(sat_pred_count_per_dc.shape[-1]) + 1) / sat_pred_count_per_dc.shape[-1]
    coverage_per_dc = (sat_pred_count_per_dc * w).sum(axis=-1) / sat_pred_count_per_dc.sum(axis=-1)

    print(f">> Mean DC violation rate (lower->better, rate is over all pairs): "
          f"{pairs_violation_rate_per_dc.mean() * 100}%")
    print(f">> Mean DC violation rate (lower->better, rate over tuples, for each tuples there exist a pair violating): "
          f"{tuple_violation_rate_per_dc.mean() * 100}%")
    print(f">> Mean Succinctness (higher->better, correlates to predicate size, higher the closer "
          f"DCs to the min-sized DC): {succinctness_per_dc.mean() * 100}%")
    print(f">> Mean Coverage (higher->better, correlates to amount of predicates being sat in DCs): "
          f"{coverage_per_dc.mean() * 100}%")

    print(f">> {(pairs_violation_rate_per_dc == 1).mean() * 100}% of the DC are perfectly satisfied :) ")
    print(f">> Worst DC, with {pairs_violation_rate_per_dc.max() * 100}% sat-rate was : "
          f"{dcs[pairs_violation_rate_per_dc.argmax()]}")

    # Save metrics:
    dc_constraints_eval = pd.DataFrame({
        'dcs_file_idx': [dc.dc_file_idx for dc in dcs],
        'dcs_repr': [str(dc) for dc in dcs],  # from which the DC can be reproduced
        'pairs_violation_rate_per_dc': pairs_violation_rate_per_dc,
        'tuple_violation_rate_per_dc': tuple_violation_rate_per_dc,
        'succinctness_per_dc': succinctness_per_dc,
        'coverage_per_dc': coverage_per_dc,
        'best_other_tuples': [json.dumps(lst) for lst in best_other_tuples_per_dc.tolist()],
        # list of the tuples with highest sat-rate in their role as 'other tuple' in the DC.
    })

    # filter DCs by their 'interesting-ness' ranking form a weighted-score of the metrics
    # adds score column, and set the order accordingly.
    # calculate additional factors
    dc_constraints_eval['tuple_violation_rate_per_dc__below_1_pct'] = dc_constraints_eval[
                                                                          'tuple_violation_rate_per_dc'] <= 0.015
    dc_constraints_eval['tuple_violation_rate_per_dc__below_5_pct'] = dc_constraints_eval[
                                                                          'tuple_violation_rate_per_dc'] <= 0.05
    dc_constraints_eval['tuple_violation_rate_per_dc__below_10_pct'] = dc_constraints_eval[
                                                                           'tuple_violation_rate_per_dc'] <= 0.10
    col_to_weight = {
        'tuple_violation_rate_per_dc__below_1_pct': 5.0,
        'tuple_violation_rate_per_dc__below_5_pct': 1.2,
        'tuple_violation_rate_per_dc__below_10_pct': 1.2,
        'pairs_violation_rate_per_dc': -1.5,
        'tuple_violation_rate_per_dc': -1.5,
        'succinctness_per_dc': 1.5,
        'coverage_per_dc': 4.0,
    }
    dc_constraints_eval['weighted_score'] = 0
    for col_name, weight in col_to_weight.items():
        dc_constraints_eval['weighted_score'] += weight * dc_constraints_eval[col_name]

    # Re-Save 'dc_constraints_eval'
    return dc_constraints_eval


def load_evaluated_dcs(eval_csv_out_path: str):
    dc_constraints_eval = pd.read_csv(eval_csv_out_path, converters={'best_other_tuples': literal_eval})

    return dc_constraints_eval
