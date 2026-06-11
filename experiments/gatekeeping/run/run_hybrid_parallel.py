"""
This script runs all hybrid scenarios in parallel. 

The SD component is run once per scenario, and its output is used to run
multiple DES trials in parallel using multiprocessing.
"""

import multiprocessing
import sys

import numpy as np
import pandas as pd

from experiment_paths import (
    THIS_DIR,
    ROOT_DIR,
    SRC_DIR,
    OUTPUT_DIR,
    RECORDS_DIR,
    METADATA_DIR,
    ARRAYS_DIR,
    SUMMARY_DIR,
)


# Make the reusable project modules importable when this file is run 
# directly.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybridsim import sd_component as sd
from hybridsim import gatekeeping_functions as gk
from hybridsim import des_component as des

from hybrid_trial_runner import (
    run_one_des_trial_from_config,
    upsert_trial_metadata,
)

des.apply_custom_record_changes()


# Simulation run controls
run_time = 365 * 5
time_points = 100000 + 1
warm_up = 90
n_trials = 10
trial_start = 10
trial_end = trial_start + n_trials
base_seed = 0
des_run_time = run_time + warm_up

severity_levels = ["low", "medium", "high"]

OVERWRITE_SD_OUTPUT = True
COMBINE_SUMMARY_FILES = True

# Set to None to choose automatically.
# For your machine with 8 logical CPUs, this will usually choose 6.
N_WORKERS = 5


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

pdfa_dir = (
    THIS_DIR
    / "../../../../../OneDrive - Cardiff University/Desktop/PhD/rsch/modelling/pattern-mining/pdfas/subspecs_length_pdfas_expo/selected"
).resolve()

PATIENT_TRIAL_ROOT = (SUMMARY_DIR / "patient_summaries_by_trial")
COHORT_TRIAL_ROOT = (SUMMARY_DIR / "cohort_summaries_by_trial")
ACTIVITY_TRIAL_ROOT = (SUMMARY_DIR / "activity_summaries_by_trial")

for directory in [
    OUTPUT_DIR,
    RECORDS_DIR,
    METADATA_DIR,
    ARRAYS_DIR,
    SUMMARY_DIR,
    PATIENT_TRIAL_ROOT,
    COHORT_TRIAL_ROOT,
    ACTIVITY_TRIAL_ROOT,
]:
    directory.mkdir(parents=True, exist_ok=True)


def get_output_paths(scenario_title):
    """
    Create and return the output paths for one scenario title.

    Parameters
    ----------
    scenario_title : str
        Title used to create the scenario-specific directories and file
        names.

    Returns
    -------
    dict
        Directories and file paths used to store scenario outputs.
    """
    scenario_title = str(scenario_title)

    record_dir = RECORDS_DIR / scenario_title
    metadata_dir = METADATA_DIR / scenario_title
    arrays_dir = ARRAYS_DIR / scenario_title

    patient_trial_dir = PATIENT_TRIAL_ROOT / scenario_title
    cohort_trial_dir = COHORT_TRIAL_ROOT / scenario_title
    activity_trial_dir = ACTIVITY_TRIAL_ROOT / scenario_title

    for directory in [
        record_dir,
        metadata_dir,
        arrays_dir,
        patient_trial_dir,
        cohort_trial_dir,
        activity_trial_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "record_dir": record_dir,
        "metadata_dir": metadata_dir,
        "arrays_dir": arrays_dir,
        "patient_trial_dir": patient_trial_dir,
        "cohort_trial_dir": cohort_trial_dir,
        "activity_trial_dir": activity_trial_dir,
        "scenario_metadata_path": (
            metadata_dir / f"hybrid_scenario_metadata_{scenario_title}.csv"
        ),
        "trial_metadata_path": (
            metadata_dir / f"hybrid_trial_metadata_{scenario_title}.csv"
        ),
        "patient_summary_path": (
            SUMMARY_DIR / f"hybrid_patient_summary_{scenario_title}.csv"
        ),
        "cohort_summary_path": (
            SUMMARY_DIR / f"hybrid_cohort_summary_{scenario_title}.csv"
        ),
        "activity_summary_path": (
            SUMMARY_DIR / f"hybrid_activity_summary_{scenario_title}.csv"
        ),
    }


# ---------------------------------------------------------------------
# SD parameters
# ---------------------------------------------------------------------

cav_population_2021 = 492315
wales_prevalence_2021 = 991216.05
wales_population_2021 = 3152120.06
wales_population_2022 = 3178152.55

wales_incidence_2022 = 231009.31

cav_population_2022 = 504723
wales_prevalence_2022 = 1009914.72

initial_population = cav_population_2021
unwell_proportion = wales_prevalence_2021 / wales_population_2021
unwell_splits = [1 - 0.0017 - 0.0011, 0.0017, 0.0011]

referral_threshold = 0.005521
presenting_proportion = ((996392.7 / 365) / (cav_population_2021 * unwell_proportion))

incidence_props = [
    (wales_incidence_2022 / ((wales_population_2021 + wales_population_2022) / 2)) / 365
]

cav_prevalence_2021 = cav_population_2021 * (
    wales_prevalence_2021 / wales_population_2021
)
cav_prevalence_2022 = cav_population_2022 * (
    wales_prevalence_2022 / wales_population_2022
)
cav_incidence_2022 = cav_population_2022 * (
    wales_incidence_2022 / wales_population_2021
)

recovery = (
    ((cav_prevalence_2021 + cav_incidence_2022 - cav_prevalence_2022) - 4700)
    / ((cav_prevalence_2021 + cav_prevalence_2022) / 2)
) / 365

sd_params = {
    "initial_population": initial_population,
    "unwell_proportion": unwell_proportion,
    "unwell_splits": unwell_splits,
    "referral_threshold": referral_threshold,
    "presenting_proportion": presenting_proportion,
    "incidence_rates": [incidence_props[-1]],
    "recovery_rates": [recovery],
    "time_points": time_points,
    "warmup_lambda_values": [7.5, 4.5, 3.0],
}


# ---------------------------------------------------------------------
# DES parameters
# ---------------------------------------------------------------------

subspecs = [
    "Foot/Ankle",
    "Hand",
    "Hip",
    "Knee",
    "Paeds",
    "Shoulder/Elbow",
    "Spine",
]

pdfa_subspec_names = [
    "foot_ankle",
    "hand",
    "hip",
    "knee",
    "paeds",
    "shoulder_elbow",
    "spine",
]

alphabet = ["A", "B", "C", "D", "E", "F", "G"]
emergency_nodes = ["B", "E", "G"]

pre_op_letter = "C"
elective_surgery_letter = "D"

subspec_probs = [
    0.084806,
    0.292071,
    0.045302,
    0.077262,
    0.386937,
    0.043835,
    0.069787,
]

subspec_probs_low = subspec_probs.copy()
subspec_probs_medium = subspec_probs.copy()
subspec_probs_high = subspec_probs.copy()

service_values = [
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

weekday_rates = [
    80.981752,
    69.324818,
    71.529197,
    59.930657,
    53.802198,
    8.926740,
    10.186813,
]

endpoints = [1, 2, 3, 4, 5, 6, 7]


# ---------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------

deterioration_functions = {
    "shift_0_025": sd.get_deterioration_rates(
        category_widths=(0.5, 0.3, 0.2),
        shift_proportion=0.025,
        shift_interval_days=182.5,
    ),
}

gatekeeping_policies = {
    "plus100pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 2.00,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "plus90pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 1.90,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "plus80pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 1.80,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "plus70pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 1.70,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "plus60pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 1.60,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "plus50pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 1.50,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "plus40pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 1.40,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "plus30pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 1.30,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "plus20pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 1.20,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "plus10pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 1.10,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "baseline": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"],
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "minus10pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 0.90,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "minus20pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 0.80,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "minus30pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 0.70,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "minus40pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 0.60,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "minus50pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 0.50,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "minus60pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 0.40,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "minus70pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 0.30,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "minus80pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 0.20,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
    "minus90pct": {
        "scenario_title": "weighted_123",
        "function": gk.weighted_priority_gatekeeping(
            threshold=sd_params["referral_threshold"] * 0.10,
            weights=np.array([1, 2, 3], dtype=float),
        ),
        "unwell_splits": [0.69767442, 0.20930233, 0.09302326],
    },
}


def run_sd_scenario(
    deterioration_function,
    gatekeeping_function,
    max_t,
    sd_params,
    unwell_splits=None,
):
    """
    Run the SD component for one hybrid scenario. 

    Parameters
    ----------
    deterioration_function : function
        The deterioration function to use in the SD model.
    gatekeeping_function : function
        The gatekeeping function to use in the SD model.
    max_t : float
        The maximum time to run the SD model for.
    sd_params : dict
        A dictionary of SD parameters to use in the SD model.
    unwell_splits : list of float, optional
        The proportions of unwell patients in each severity category. If
        None, the value from sd_params["unwell_splits"] will be used.

    Returns
    -------
    dict
        The solved SD model, time points, stock values, referral rates, and
        severity split used for the scenario.    
    """
    if unwell_splits is None:
        unwell_splits = sd_params["unwell_splits"]

    model = sd.SD(
        population_function=sd.get_time_dependent_population_size(
            [sd_params["initial_population"]],
            max_t,
        ),
        initial_unwell_proportion=sd_params["unwell_proportion"],
        unwell_splits=unwell_splits,
        gatekeeping_function=gatekeeping_function,
        presenting_proportion=sd_params["presenting_proportion"],
        deterioration_function=deterioration_function,
        incidence_function=sd.get_time_dependent_incidence_rate(
            sd_params["incidence_rates"],
            max_t,
        ),
        recovery_function=sd.get_time_dependent_recovery_rate(
            sd_params["recovery_rates"],
            max_t,
        ),
    )

    t = np.linspace(0, max_t, sd_params["time_points"])
    model.solve(t=t)

    return {
        "model": model,
        "t": t,
        "stocks": np.asarray(model.P).T,
        "lambdas": np.asarray(model.lambdas).T,
        "unwell_splits": np.asarray(unwell_splits, dtype=float),
    }


def validate_scenario_indices(metadata, source_description):
    """
    Check that each scenario_file has only one scenario_index.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Metadata containing ``scenario_file`` and ``scenario_index``.
    source_description : str
        Description used to identify the metadata in error messages.
    
    Raises
    ------
    ValueError
        If required values are missing or invalid, one scenario has
        multiple indices, or one index belongs to multiple scenarios.
    """
    required_columns = {"scenario_file", "scenario_index"}

    if not required_columns.issubset(metadata.columns):
        missing_columns = required_columns - set(metadata.columns)
        raise ValueError(
            f"{source_description} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    metadata_to_check = metadata[
        ["scenario_file", "scenario_index"]
    ].copy()

    metadata_to_check["scenario_index"] = pd.to_numeric(
        metadata_to_check["scenario_index"],
        errors="coerce",
    )

    metadata_to_check = metadata_to_check.dropna(
        subset=["scenario_file", "scenario_index"]
    )

    index_counts = (
        metadata_to_check
        .groupby("scenario_file")["scenario_index"]
        .nunique()
    )

    conflicting_scenarios = index_counts[index_counts > 1]

    if not conflicting_scenarios.empty:
        raise ValueError(
            f"Some scenarios in {source_description} have been assigned "
            "multiple scenario_index values: "
            f"{conflicting_scenarios.index.to_list()}"
        )


def prepare_scenario_metadata():
    """
    Build current scenarios and update scenario metadata files.

    Existing scenario indices are retained where available, and
    sequential indices are assigned to new scenarios. Combined and
    policy-specific scenario metadata files are then written to disk.

    Returns
    -------
    tuple of pandas.DataFrame
        The current scenarios and the combined scenario metadata.
    """
    scenario_rows = [] 

    for deterioration_label, deterioration_function in deterioration_functions.items():
        for policy_name, policy_config in gatekeeping_policies.items():
            scenario_rows.append(
                {
                    "policy": policy_config["scenario_title"],
                    "scenario": policy_name,
                    "scenario_file": f"{policy_config['scenario_title']}_{policy_name}",    
                    "deterioration_label": deterioration_label,
                    "unwell_splits": policy_config["unwell_splits"],
                    "gatekeeping_function": policy_config["function"],
                    "deterioration_function": deterioration_function,
                }
            )

    scenarios = pd.DataFrame(scenario_rows) 

    metadata_columns = [
        "scenario_index",
        "policy",
        "scenario",
        "scenario_file",
        "deterioration_label",
        "unwell_splits",
    ]

    scenario_metadata_path = METADATA_DIR / "hybrid_scenario_metadata.csv"

    if scenario_metadata_path.exists():
        existing_scenario_metadata = pd.read_csv(scenario_metadata_path)
    else:
        existing_scenario_metadata = pd.DataFrame(columns=metadata_columns)

    validate_scenario_indices(
        metadata=existing_scenario_metadata,
        source_description="the existing scenario metadata",
    )

    existing_scenario_metadata = (
        existing_scenario_metadata
        .drop_duplicates(subset=["scenario_file"], keep="last")
        .copy()
    )

    existing_scenario_metadata["scenario_index"] = pd.to_numeric(
        existing_scenario_metadata["scenario_index"],
        errors="coerce",
    )

    # Preserve established indices across separate experiment runs.
    existing_index_map = (
        existing_scenario_metadata
        .dropna(subset=["scenario_file", "scenario_index"])
        .set_index("scenario_file")["scenario_index"]
        .to_dict()
    )
    scenarios["scenario_index"] = scenarios["scenario_file"].map(
        existing_index_map
    )

    if len(existing_index_map) > 0:
        next_scenario_index = int(max(existing_index_map.values())) + 1
    else:
        next_scenario_index = 0

    needs_assignment = scenarios["scenario_index"].isna()
    n_new_scenarios = needs_assignment.sum()
    scenarios.loc[needs_assignment, "scenario_index"] = range(
        next_scenario_index,
        next_scenario_index + n_new_scenarios,
    )
    scenarios["scenario_index"] = scenarios["scenario_index"].astype(int)

    scenarios = (
        scenarios
        .sort_values(["scenario_index", "scenario_file"])
        .reset_index(drop=True)
    )
    current_scenario_metadata = scenarios[metadata_columns].copy()

    combined_scenario_metadata = pd.concat(
        [existing_scenario_metadata, current_scenario_metadata],
        ignore_index=True,
    )

    validate_scenario_indices(
        metadata=combined_scenario_metadata,
        source_description="the combined scenario metadata",
    )

    scenario_metadata = (
        combined_scenario_metadata
        .drop_duplicates(
            subset=["scenario_file"], 
            keep="last",
        )
        .sort_values(["scenario_index", "scenario_file"])
        .reset_index(drop=True)
    )
    scenario_metadata["scenario_index"] = scenario_metadata[
        "scenario_index"
    ].astype(int)

    scenario_metadata.to_csv(scenario_metadata_path, index=False)

    grouped_metadata = scenario_metadata.groupby("policy")
    for scenario_title, scenario_metadata_group in grouped_metadata:
        paths = get_output_paths(scenario_title)
        scenario_metadata_group.to_csv(
            paths["scenario_metadata_path"],
            index=False,
        )

    return scenarios, scenario_metadata


def build_scenario_index_lookup(scenario_metadata):
    """
    Build a scenario-to-index lookup from existing metadata.

    The lookup includes scenario metadata and any existing trial metadata
    files. This keeps scenario indices consistent when new scenarios have
    been added since the trial metadata was last updated.

    Parameters
    ----------
    scenario_metadata : pandas.DataFrame
        Scenario metadata containing ``scenario_file`` and
        ``scenario_index``.

    Returns
    -------
    pandas.DataFrame
        A validated lookup with one index for each scenario.
    """
    lookup_frames = [
        scenario_metadata[["scenario_file", "scenario_index"]]
    ]

    trial_metadata_paths = sorted(
        METADATA_DIR.glob("*/hybrid_trial_metadata_*.csv")
    )

    for trial_metadata_path in trial_metadata_paths:
        existing_trial_metadata = pd.read_csv(trial_metadata_path)

        if {"scenario_file", "scenario_index"}.issubset(existing_trial_metadata.columns):
            lookup_frames.append(
                existing_trial_metadata[["scenario_file", "scenario_index"]]
            )

    scenario_index_lookup = pd.concat(
        lookup_frames,
        ignore_index=True,
    )

    scenario_index_lookup["scenario_index"] = pd.to_numeric(
        scenario_index_lookup["scenario_index"],
        errors="coerce",
    )

    scenario_index_lookup = scenario_index_lookup.dropna(
        subset=["scenario_file", "scenario_index"]
    )

    validate_scenario_indices(
        metadata=scenario_index_lookup,
        source_description="the scenario and trial metadata",
    )

    scenario_index_lookup = (
        scenario_index_lookup
        .drop_duplicates(subset=["scenario_file"], keep="last")
        .reset_index(drop=True)
    )

    scenario_index_lookup["scenario_index"] = (
        scenario_index_lookup["scenario_index"].astype(int)
    )

    return scenario_index_lookup


def choose_n_workers(n_trials):
    """
    Choose a conservative number of worker processes. 
    
    This automatic choice is used when the module-level N_WORKERS setting 
    is None. The result is limited to the number of requested trials.
    """
    cpu_count = multiprocessing.cpu_count()
    return min(n_trials, max(1, cpu_count - 2))


def validate_run_configuration():
    """
    Validate configuration values and positional parameter relationships.
    """
    if run_time <= 0:
        raise ValueError("run_time must be greater than zero.")

    if time_points < 2:
        raise ValueError("time_points must be at least 2.")

    if warm_up < 0:
        raise ValueError("warm_up cannot be negative.")

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    if trial_start < 0:
        raise ValueError("trial_start cannot be negative.")

    if trial_end != trial_start + n_trials:
        raise ValueError(
            "trial_end must equal trial_start + n_trials."
        )

    if N_WORKERS is not None and int(N_WORKERS) < 1:
        raise ValueError("N_WORKERS must be at least 1 or None.")

    if not pdfa_dir.is_dir():
        raise FileNotFoundError(
            f"PDFA directory does not exist: {pdfa_dir}"
        )

    aligned_lengths = {
        len(subspecs),
        len(pdfa_subspec_names),
        len(service_values),
        len(subspec_probs),
    }
    if len(aligned_lengths) != 1:
        raise ValueError(
            "subspecs, pdfa_subspec_names, service_values, and "
            "subspec_probs must have the same length."
        )

    if any(len(values) != len(alphabet) for values in service_values):
        raise ValueError(
            "Each service_values row must contain one value per activity."
        )

    if any(value < 0 for values in service_values for value in values):
        raise ValueError("service_values cannot contain negative values.")

    if any(probability < 0 for probability in subspec_probs):
        raise ValueError("subspec_probs cannot contain negative values.")

    if not np.isclose(sum(subspec_probs), 1.0):
        raise ValueError("subspec_probs must sum to 1.")

    if len(weekday_rates) != len(endpoints):
        raise ValueError(
            "weekday_rates and endpoints must have the same length."
        )

    if any(rate < 0 for rate in weekday_rates):
        raise ValueError("weekday_rates cannot contain negative values.")

    if len(deterioration_functions) != 1:
        raise ValueError(
            "Scenario filenames do not currently distinguish between "
            "multiple deterioration functions."
        )

    for policy_name, policy_config in gatekeeping_policies.items():
        splits = np.asarray(policy_config["unwell_splits"], dtype=float)

        if splits.shape != (len(severity_levels),):
            raise ValueError(
                f"Policy {policy_name!r} must define one unwell split "
                "per severity level."
            )

        if np.any(splits < 0) or not np.isclose(splits.sum(), 1.0):
            raise ValueError(
                f"Policy {policy_name!r} has invalid unwell_splits."
            )


def read_csv_with_diagnostics(file):
    """
    Read a CSV file and print the path if reading fails.
    """
    try:
        return pd.read_csv(file)
    except Exception as error:
        print()
        print("Failed to read CSV file:")
        print(file)
        print(f"Path length: {len(str(file))}")
        print(f"Error: {repr(error)}")
        raise


def combine_csv_files(files, output_path):
    """
    Combine CSV files and write them to the specified output path.
    """
    if not files:
        return

    dataframes = [
        read_csv_with_diagnostics(file)
        for file in files
    ]

    pd.concat(
        dataframes,
        ignore_index=True,
    ).to_csv(
        output_path,
        index=False,
    )


def combine_trial_summary_files(scenarios):
    """
    Combine per-trial summary files into one file per scenario title.
    """
    summary_specs = [
        (
            "patient_trial_dir",
            "*_patient_summary.csv",
            "patient_summary_path",
        ),
        (
            "cohort_trial_dir",
            "*_cohort_summary.csv",
            "cohort_summary_path",
        ),
        (
            "activity_trial_dir",
            "*_activity_summary.csv",
            "activity_summary_path",
        ),
    ]

    for policy in sorted(scenarios["policy"].unique()):
        paths = get_output_paths(policy)

        for directory_key, pattern, output_key in summary_specs:
            files = sorted(paths[directory_key].glob(pattern))

            combine_csv_files(
                files=files,
                output_path=paths[output_key],
            )


def main():
    """
    Run all hybrid scenarios.

    The SD component is run once per scenario. The DES trials for that
    scenario are then submitted to a multiprocessing pool.
    """
    validate_run_configuration()

    if N_WORKERS is None:
        n_workers = choose_n_workers(n_trials)
    else:
        n_workers = int(N_WORKERS)
    print(f"CPU count visible to Python: {multiprocessing.cpu_count()}")
    print(f"Using {n_workers} worker process(es)")

    scenarios, scenario_metadata = prepare_scenario_metadata()
    scenario_index_lookup = build_scenario_index_lookup(scenario_metadata)

    if "policy" not in scenarios.columns:
        raise ValueError(
            "The scenarios DataFrame must contain a 'policy' column. "
            "Add this when constructing scenarios from gatekeeping_policies."
        )

    pdfa_lookup, alphabet_lookup = des.load_pdfa_lookup(
        pdfa_subspec_names=pdfa_subspec_names,
        severity_levels=severity_levels,
        pdfa_dir=pdfa_dir,
    )

    pdfas, alphabets = des.get_pdfa_lists(
        pdfa_lookup=pdfa_lookup,
        alphabet_lookup=alphabet_lookup,
        pdfa_subspec_names=pdfa_subspec_names,
        severity_levels=severity_levels,
    )

    activity_dict, _ = des.get_activity_dictionaries(
        alphabet=alphabet,
        start_value=3,
    )

    nodes = des.get_list_of_nodes(alphabets, subspecs)

    scenario_index_lookup_records = scenario_index_lookup.to_dict("records")

    for _, scenario_row in scenarios.iterrows():
        scenario_name = scenario_row["scenario_file"]
        scenario_title = scenario_row["policy"]
        scenario_index = int(scenario_row["scenario_index"])

        print()
        print(f"Running trials for scenario: {scenario_name}")
        print(f"  Saving under scenario title: {scenario_title}")

        paths = get_output_paths(scenario_title)

        record_dir = paths["record_dir"]
        arrays_dir = paths["arrays_dir"]
        patient_trial_dir = paths["patient_trial_dir"]
        cohort_trial_dir = paths["cohort_trial_dir"]
        activity_trial_dir = paths["activity_trial_dir"]
        trial_metadata_path = paths["trial_metadata_path"]

        sd_file_name = f"{scenario_name}_sd_output.npz"
        sd_output_path = arrays_dir / sd_file_name

        sd_result = run_sd_scenario(
            deterioration_function=scenario_row["deterioration_function"],
            gatekeeping_function=scenario_row["gatekeeping_function"],
            max_t=run_time,
            sd_params=sd_params,
            unwell_splits=scenario_row["unwell_splits"],
        )

        stocks_original = sd_result["stocks"]
        lambdas_original = sd_result["lambdas"]
        t_original = sd_result["t"]

        lambdas_with_warmup, t_with_warmup = sd.add_constant_lambda_warmup(
            lambdas=lambdas_original.T,
            ts=t_original,
            warmup_days=warm_up,
            value="initial",
            shift_time=True,
        )

        lambdas_with_warmup = lambdas_with_warmup.T

        if sd_output_path.exists() and not OVERWRITE_SD_OUTPUT:
            print(
                f"  Skipping SD output save for {scenario_name}; "
                "Array file already exists."
            )
        else:
            np.savez(
                sd_output_path,
                t=t_original,
                t_with_warmup=t_with_warmup,
                stocks=stocks_original,
                lambdas=lambdas_original,
                lambdas_with_warmup=lambdas_with_warmup,
                sd_severity_levels=np.asarray(severity_levels),
                scenario_name=scenario_name,
                scenario_index=scenario_index,
                scenario_title=scenario_title,
                unwell_splits=np.asarray(
                    scenario_row["unwell_splits"], dtype=float,
                ),
            )

        trial_configs = []

        for trial in range(trial_start, trial_end):
            seed = base_seed + trial

            records_filename = f"{scenario_name}_trial_{trial:03d}_records.csv"
            patient_filename = f"{scenario_name}_trial_{trial:03d}_patient_summary.csv"
            cohort_filename = f"{scenario_name}_trial_{trial:03d}_cohort_summary.csv"
            activity_filename = f"{scenario_name}_trial_{trial:03d}_activity_summary.csv"

            records_path = record_dir / records_filename
            patient_path = patient_trial_dir / patient_filename
            cohort_path = cohort_trial_dir / cohort_filename
            activity_path = activity_trial_dir / activity_filename

            if (
                records_path.exists()
                and patient_path.exists()
                and cohort_path.exists()
                and activity_path.exists()
            ):
                print(f"  Skipping completed trial {trial}; records already exist.")
                continue

            trial_configs.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_title": scenario_title,
                    "scenario_index": scenario_index,
                    "trial": trial,
                    "seed": seed,
                    "lambdas_with_warmup": lambdas_with_warmup,
                    "t_with_warmup": t_with_warmup,
                    "run_time": run_time,
                    "des_run_time": des_run_time,
                    "warm_up": warm_up,
                    "records_path": str(records_path),
                    "patient_path": str(patient_path),
                    "cohort_path": str(cohort_path),
                    "activity_path": str(activity_path),
                    "records_file_relative": str(records_path.relative_to(OUTPUT_DIR)),
                    "patient_file_relative": str(patient_path.relative_to(OUTPUT_DIR)),
                    "cohort_file_relative": str(cohort_path.relative_to(OUTPUT_DIR)),
                    "activity_file_relative": str(activity_path.relative_to(OUTPUT_DIR)),
                    "weekday_rates": weekday_rates,
                    "endpoints": endpoints,
                    "pdfas": pdfas,
                    "alphabets": alphabets,
                    "subspecs": subspecs,
                    "service_values": service_values,
                    "emergency_nodes": emergency_nodes,
                    "subspec_probs_low": subspec_probs_low,
                    "subspec_probs_medium": subspec_probs_medium,
                    "subspec_probs_high": subspec_probs_high,
                    "activity_dict": activity_dict,
                    "nodes": nodes,
                    "pre_op_letter": pre_op_letter,
                    "elective_surgery_letter": elective_surgery_letter,
                    "scenario_index_lookup_records": scenario_index_lookup_records,
                }
            )

        if len(trial_configs) == 0:
            print(f"  All DES trials already complete for {scenario_name}.")
            continue

        workers_for_this_scenario = min(n_workers, len(trial_configs))

        print(
            f"  Running {len(trial_configs)} DES trial(s) using "
            f"{workers_for_this_scenario} worker process(es)"
        )

        with multiprocessing.Pool(
            processes=workers_for_this_scenario
        ) as pool:
            trial_rows = pool.map(run_one_des_trial_from_config, trial_configs)

        for trial_row in trial_rows:
            upsert_trial_metadata(
                row=trial_row,
                metadata_path=trial_metadata_path,
                scenario_index_lookup=scenario_index_lookup,
            )

        print(f"  Saved metadata for {scenario_name}")

        print(f"Completed all trials for Scenario: {scenario_name}")

    if COMBINE_SUMMARY_FILES:
        combine_trial_summary_files(scenarios)
    else:
        print(" Summary file combination is disabled. Recommend reviewing "
                "and merging summary files separately.")

    print()
    print("All requested scenarios and trials are complete.")


if __name__ == "__main__":
    main()