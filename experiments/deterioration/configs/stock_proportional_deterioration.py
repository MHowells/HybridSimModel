"""
Configuration for the stock proportional deterioration experiment.
"""

import numpy as np

from hybridsim import gatekeeping_functions as gk

from configs import common


EXPERIMENT_NAME = "stock_proportional_weighted_equal"
GATEKEEPING_POLICY = "weighted_equal"

GATEKEEPING_FUNCTION = gk.weighted_priority_gatekeeping(
    threshold=common.REFERRAL_THRESHOLD,
    weights=[1, 1, 1],
)

STOCK_PROPORTIONAL_MEAN_TIMES = {
    "six_months": 180,
    "two_years": 730,
    "five_years": 1825,
    "ten_years": 3650,
}


def constant_deterioration_rate(mean_time_days):
    """
    Return a constant stock proportional deterioration-rate function.

    Parameters
    ----------
    mean_time_days : float
        Mean time before deterioration, in days.

    Returns
    -------
    function
        Function accepting simulation time and returning a constant daily
        deterioration rate.
    """
    if mean_time_days <= 0:
        raise ValueError(
            "mean_time_days must be greater than zero."
        )

    daily_rate = 1 / mean_time_days

    def deterioration_function(t):
        return daily_rate

    return deterioration_function


def build_scenario_definitions():
    """
    Construct the stock proportional deterioration scenarios.

    Returns
    -------
    list of dict
        Scenario metadata and callable model functions.
    """
    scenario_definitions = []

    for label, mean_time_days in STOCK_PROPORTIONAL_MEAN_TIMES.items():
        scenario_definitions.append(
            {
                "scenario": (
                    f"stock_proportional_{label}_{GATEKEEPING_POLICY}"
                ),
                "deterioration_model": "stock_proportional",
                "deterioration_label": label,
                "stock_proportional_mean_time_days": mean_time_days,
                "boundary_shift_proportion": np.nan,
                "gatekeeping_policy": GATEKEEPING_POLICY,
                "description": (
                    "Stock proportional deterioration: "
                    f"{label}; gatekeeping: {GATEKEEPING_POLICY}"
                ),
                "deterioration_function": (
                    constant_deterioration_rate(mean_time_days)
                ),
                "gatekeeping_function": GATEKEEPING_FUNCTION,
            }
        )

    return scenario_definitions


SCENARIO_DEFINITIONS = build_scenario_definitions()