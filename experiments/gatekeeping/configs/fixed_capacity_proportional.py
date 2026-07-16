"""Configuration for the fixed capacity gatekeeping scenarios."""

import gatekeeping_functions as gk
import sd_component as sd

from configs.common import SD_PARAMS


SCENARIO_TITLE = "proportional_fixed_capacity"

UNWELL_SPLITS = [
    0.5, 
    0.3, 
    0.2,
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
    "03": 3.00,
    "06": 6.00,
    "09": 9.00,
    "12": 12.00,
    "15": 15.00,
    "18": 18.00,
    "21": 21.00,
    "24": 24.00,
    "27": 27.00,
    "30": 30.00,
}


GATEKEEPING_POLICIES = {
    scenario_name: {
        "scenario_title": SCENARIO_TITLE,
        "function": gk.fixed_capacity_proportional_gatekeeping(
            capacity=referral_number,
        ),
        "unwell_splits": UNWELL_SPLITS.copy(),
    }
    for scenario_name, referral_number
    in REFERRAL_NUMBERS.items()
}
