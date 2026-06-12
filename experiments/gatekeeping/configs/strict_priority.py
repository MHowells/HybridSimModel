"""Configuration for the strict priority gatekeeping scenarios."""

from hybridsim import gatekeeping_functions as gk
from hybridsim import sd_component as sd

from configs.common import SD_PARAMS


SCENARIO_TITLE = "strict_priority"

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
REFERRAL_MULTIPLIER = {
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
    "minus100pct": 0.00,
}


GATEKEEPING_POLICIES = {
    scenario_name: {
        "scenario_title": SCENARIO_TITLE,
        "function": gk.strict_priority_gatekeeping(
            threshold=(
                SD_PARAMS["referral_threshold"]
                * referral_multiplier,
            ),
        ),
        "unwell_splits": UNWELL_SPLITS.copy(),
    }
    for scenario_name, referral_multiplier
    in REFERRAL_MULTIPLIER.items()
}
