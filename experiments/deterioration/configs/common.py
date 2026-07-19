"""
Shared configuration for the SD deterioration exploration.
"""


# Simulation controls.

RUN_TIME = 365 * 5
TIME_POINTS = 100000 + 1

SEVERITY_LEVELS = [
    "low",
    "medium",
    "high",
]

# A long-format CSV contains one row per scenario and time point. The NPZ
# output is substantially smaller.
WRITE_TIMESERIES_CSV = False


# Population and epidemiological inputs.

CAV_POPULATION_2021 = 492315
CAV_POPULATION_2022 = 504723

WALES_POPULATION_2021 = 3152120.06
WALES_POPULATION_2022 = 3178152.55

WALES_PREVALENCE_2021 = 991216.05
WALES_PREVALENCE_2022 = 1009914.72

WALES_INCIDENCE_2022 = 231009.31


# Derived SD parameters.

INITIAL_POPULATION = CAV_POPULATION_2021

UNWELL_PROPORTION = (
    WALES_PREVALENCE_2021
    / WALES_POPULATION_2021
)

UNWELL_SPLITS = [
    0.50,
    0.30,
    0.20,
]

REFERRAL_THRESHOLD = 0.005521

PRESENTING_PROPORTION = (
    (996392.7 / 365)
    / (CAV_POPULATION_2021 * UNWELL_PROPORTION)
)

INCIDENCE_PROPORTION = (
    WALES_INCIDENCE_2022
    / (
        (
            WALES_POPULATION_2021
            + WALES_POPULATION_2022
        )
        / 2
    )
) / 365

CAV_PREVALENCE_2021 = CAV_POPULATION_2021 * (
    WALES_PREVALENCE_2021
    / WALES_POPULATION_2021
)

CAV_PREVALENCE_2022 = CAV_POPULATION_2022 * (
    WALES_PREVALENCE_2022
    / WALES_POPULATION_2022
)

CAV_INCIDENCE_2022 = CAV_POPULATION_2022 * (
    WALES_INCIDENCE_2022
    / WALES_POPULATION_2021
)

RECOVERY = (
    (
        CAV_PREVALENCE_2021
        + CAV_INCIDENCE_2022
        - CAV_PREVALENCE_2022
        - 4700
    )
    / (
        (
            CAV_PREVALENCE_2021
            + CAV_PREVALENCE_2022
        )
        / 2
    )
) / 365


SD_PARAMS = {
    "initial_population": INITIAL_POPULATION,
    "unwell_proportion": UNWELL_PROPORTION,
    "unwell_splits": UNWELL_SPLITS,
    "referral_threshold": REFERRAL_THRESHOLD,
    "presenting_proportion": PRESENTING_PROPORTION,
    "incidence_rates": [INCIDENCE_PROPORTION],
    "recovery_rates": [RECOVERY],
    "time_points": TIME_POINTS,
}
