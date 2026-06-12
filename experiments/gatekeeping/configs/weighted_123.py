"""Configuration for the weighted 1:2:3 gatekeeping scenarios."""

import numpy as np

from hybridsim import gatekeeping_functions as gk
from hybridsim import sd_component as sd

from configs.common import SD_PARAMS


SCENARIO_TITLE = "weighted_123"

UNWELL_SPLITS = [
    0.69767442,
    0.20930233,
    0.09302326,
]

WEIGHTS = np.array(
    [1.0, 2.0, 3.0],
    dtype=float,
)

DETERIORATION_FUNCTIONS = {
    "shift_0_025": sd.get_deterioration_rates(
        category_widths=(0.5, 0.3, 0.2),
        shift_proportion=0.025,
        shift_interval_days=182.5,
    ),
}

# Dictionary order determines the order in which scenarios are built.
THRESHOLD_MULTIPLIERS = {
    "plus100pct": 2.00,
    "plus90pct": 1.90,
    "plus80pct": 1.80,
    "plus70pct": 1.70,
    "plus60pct": 1.60,
    "plus50pct": 1.50,
    "plus40pct": 1.40,
    "plus30pct": 1.30,
    "plus20pct": 1.20,
    "plus10pct": 1.10,
    "baseline": 1.00,
    "minus10pct": 0.90,
    "minus20pct": 0.80,
    "minus30pct": 0.70,
    "minus40pct": 0.60,
    "minus50pct": 0.50,
    "minus60pct": 0.40,
    "minus70pct": 0.30,
    "minus80pct": 0.20,
    "minus90pct": 0.10,
}


GATEKEEPING_POLICIES = {
    scenario_name: {
        "scenario_title": SCENARIO_TITLE,
        "function": gk.weighted_priority_gatekeeping(
            threshold=(
                SD_PARAMS["referral_threshold"]
                * threshold_multiplier
            ),
            weights=WEIGHTS.copy(),
        ),
        "unwell_splits": UNWELL_SPLITS.copy(),
    }
    for scenario_name, threshold_multiplier
    in THRESHOLD_MULTIPLIERS.items()
}
