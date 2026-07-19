"""
Configuration for the boundary shift deterioration experiment.
"""

import numpy as np

from hybridsim import gatekeeping_functions as gk
from hybridsim import sd_component as sd

from configs import common


EXPERIMENT_NAME = "boundary_shift_weighted_equal"
GATEKEEPING_POLICY = "weighted_equal"

GATEKEEPING_FUNCTION = gk.weighted_priority_gatekeeping(
    threshold=common.REFERRAL_THRESHOLD,
    weights=[1, 1, 1],
)

BOUNDARY_SHIFT_PROPORTIONS = [
    0.01,
    0.02,
    0.025,
    0.05,
    0.075,
    0.10,
]

BOUNDARY_SHIFT_CATEGORY_WIDTHS = (
    0.50,
    0.30,
    0.20,
)

BOUNDARY_SHIFT_INTERVAL_DAYS = 182.5


def build_scenario_definitions():
    """
    Construct the boundary shift deterioration scenarios.

    Returns
    -------
    list of dict
        Scenario metadata and callable model functions.
    """
    scenario_definitions = []

    for shift_proportion in BOUNDARY_SHIFT_PROPORTIONS:
        deterioration_function = sd.get_deterioration_rates(
            category_widths=BOUNDARY_SHIFT_CATEGORY_WIDTHS,
            shift_proportion=shift_proportion,
            shift_interval_days=(
                BOUNDARY_SHIFT_INTERVAL_DAYS
            ),
        )

        scenario_definitions.append(
            {
                "scenario": (
                    "boundary_shift_"
                    f"{str(shift_proportion).replace('.', '_')}_"
                    f"{GATEKEEPING_POLICY}"
                ),
                "deterioration_model": "boundary_shift",
                "deterioration_label": (
                    f"shift_{shift_proportion:g}"
                ),
                "whole_stock_mean_time_days": np.nan,
                "boundary_shift_proportion": shift_proportion,
                "gatekeeping_policy": GATEKEEPING_POLICY,
                "description": (
                    "Boundary shift deterioration: "
                    f"{shift_proportion:g} shift per six months; "
                    f"gatekeeping: {GATEKEEPING_POLICY}"
                ),
                "deterioration_function": deterioration_function,
                "gatekeeping_function": GATEKEEPING_FUNCTION,
            }
        )

    return scenario_definitions


SCENARIO_DEFINITIONS = build_scenario_definitions()