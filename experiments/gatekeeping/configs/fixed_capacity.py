"""Configuration for the weighted 1:2:3 gatekeeping scenarios."""

import numpy as np

import gatekeeping_functions as gk
import sd_component as sd

from configs.common import SD_PARAMS


SCENARIO_TITLE = "fixed_capacity"

UNWELL_SPLITS = [
    1 - 0.0017 - 0.0011, 
    0.0017, 
    0.0011
]

DETERIORATION_FUNCTIONS = {
    "shift_0_025": sd.get_deterioration_rates(
        category_widths=(0.5, 0.3, 0.2),
        shift_proportion=0.025,
        shift_interval_days=182.5,
    ),
}

# Dictionary order determines the order in which scenarios are built.
REFERRAL_NUMBERS = {
    "03_strict": 3.00,
    "06_strict": 6.00,
    "09_strict": 9.00,
    "12_strict": 12.00,
    "15_strict": 15.00,
    "18_strict": 18.00,
    "21_strict": 21.00,
    "24_strict": 24.00,
    "27_strict": 27.00,
    "30_strict": 30.00,
}


GATEKEEPING_POLICIES = {
    scenario_name: {
        "scenario_title": SCENARIO_TITLE,
        "function": gk.fixed_capacity_strict_gatekeeping(
            capacity=referral_number,
        ),
        "unwell_splits": UNWELL_SPLITS.copy(),
    }
    for scenario_name, referral_number
    in REFERRAL_NUMBERS.items()
}
