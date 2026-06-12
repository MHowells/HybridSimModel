"""
Shared configuration for all gatekeeping hybrid scenarios.

This module contains simulation controls and SD/DES parameters that are
intended to remain the same across scenario families.
"""

from run.experiment_paths import THIS_DIR


# Simulation run controls
RUN_TIME = 365 * 5
TIME_POINTS = 100000 + 1
WARM_UP = 90

N_TRIALS = 10
TRIAL_START = 10
TRIAL_END = TRIAL_START + N_TRIALS
BASE_SEED = 0

DES_RUN_TIME = RUN_TIME + WARM_UP

SEVERITY_LEVELS = ["low", "medium", "high"]

OVERWRITE_SD_OUTPUT = True
COMBINE_SUMMARY_FILES = True

# Set to None to choose automatically based on the number cores.
N_WORKERS = 5


# Define paths
PDFA_DIR = (
    THIS_DIR
    / "../../../../../OneDrive - Cardiff University/Desktop/PhD/"
    "rsch/modelling/pattern-mining/pdfas/"
    "subspecs_length_pdfas_expo/selected"
).resolve()


# Shared SD parameters
CAV_POPULATION_2021 = 492315
CAV_POPULATION_2022 = 504723

WALES_POPULATION_2021 = 3152120.06
WALES_POPULATION_2022 = 3178152.55

WALES_PREVALENCE_2021 = 991216.05
WALES_PREVALENCE_2022 = 1009914.72

WALES_INCIDENCE_2022 = 231009.31

INITIAL_POPULATION = CAV_POPULATION_2021

UNWELL_PROPORTION = (
    WALES_PREVALENCE_2021
    / WALES_POPULATION_2021
)

UNWELL_SPLITS = [
    1 - 0.0017 - 0.0011,
    0.0017,
    0.0011,
]

REFERRAL_THRESHOLD = 0.005521

PRESENTING_PROPORTION = (
    (996392.7 / 365)
    / (CAV_POPULATION_2021 * UNWELL_PROPORTION)
)

INCIDENCE_PROPORTIONS = [
    (
        WALES_INCIDENCE_2022
        / (
            (
                WALES_POPULATION_2021
                + WALES_POPULATION_2022
            )
            / 2
        )
    )
    / 365
]

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

# "warmup_lambda_values" are defined based on initial splits per-scenario later.
# These are default values that will be used if a scenario doesn't
# specify its own warmup_lambda_values.
SD_PARAMS = {
    "initial_population": INITIAL_POPULATION,
    "unwell_proportion": UNWELL_PROPORTION,
    "unwell_splits": UNWELL_SPLITS,
    "referral_threshold": REFERRAL_THRESHOLD,
    "presenting_proportion": PRESENTING_PROPORTION,
    "incidence_rates": INCIDENCE_PROPORTIONS,
    "recovery_rates": [RECOVERY],
    "time_points": TIME_POINTS,
    "warmup_lambda_values": [7.5, 4.5, 3.0],
}


# Shared DES parameters
# The ordering of subspecs, pdfa_subspec_names, and service_values must 
# be consistent as they are used together to construct the PDFAs and DES
# parameters.
SUBSPECS = [
    "Foot/Ankle",
    "Hand",
    "Hip",
    "Knee",
    "Paeds",
    "Shoulder/Elbow",
    "Spine",
]

PDFA_SUBSPEC_NAMES = [
    "foot_ankle",
    "hand",
    "hip",
    "knee",
    "paeds",
    "shoulder_elbow",
    "spine",
]

ALPHABET = ["A", "B", "C", "D", "E", "F", "G"]

EMERGENCY_NODES = ["B", "E", "G"]

PRE_OP_LETTER = "C"
ELECTIVE_SURGERY_LETTER = "D"

SUBSPEC_PROBS = [
    0.084806,
    0.292071,
    0.045302,
    0.077262,
    0.386937,
    0.043835,
    0.069787,
]

SUBSPEC_PROBS_LOW = SUBSPEC_PROBS.copy()
SUBSPEC_PROBS_MEDIUM = SUBSPEC_PROBS.copy()
SUBSPEC_PROBS_HIGH = SUBSPEC_PROBS.copy()

SERVICE_VALUES = [
    [
        365 / 1220,
        0,
        365 / 585,
        365 / 458.9,
        0,
        365 / 1280,
        0,
    ],
    [
        365 / 1277,
        0,
        365 / 1397,
        365 / 923.4,
        0,
        365 / 1636,
        0,
    ],
    [
        365 / 1699.075,
        0,
        365 / 1135.4875,
        365 / 641.3,
        0,
        365 / 1758.9,
        0,
    ],
    [
        365 / 1366.575,
        0,
        365 / 882.7875,
        365 / 1225.5,
        0,
        365 / 1157.1,
        0,
    ],
    [
        365 / 1024,
        0,
        365 / 140,
        365 / 427.5,
        0,
        365 / 2873,
        0,
    ],
    [
        365 / 1097,
        0,
        365 / 438.9,
        365 / 259.4,
        0,
        365 / 758.1,
        0,
    ],
    [
        365 / 2081,
        0,
        365 / 692,
        365 / 669.8,
        0,
        365 / 3092,
        0,
    ],
]

WEEKDAY_RATES = [
    80.981752,
    69.324818,
    71.529197,
    59.930657,
    53.802198,
    8.926740,
    10.186813,
]

ENDPOINTS = [1, 2, 3, 4, 5, 6, 7]
