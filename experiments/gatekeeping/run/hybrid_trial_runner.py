"""
This module contains the functions required to run one DES trial for one
hybrid scenario, and to write the results to CSV files. 
"""

from pathlib import Path
import sys

import ciw
import pandas as pd

from experiment_paths import SRC_DIR

# Make the reusable project modules importable when this file is run
# directly.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybridsim import des_component as des
from hybridsim import results as res

des.apply_custom_record_changes()


def write_csv_safely(
    dataframe, 
    path, 
    index=False, 
    compression=None,
):
    """
    Write a CSV via a temporary file before replacing the target file.

    This reduces the chance of being left with a partially-written CSV if
    the run is interrupted.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Data to write.
    path : path-like
        Destination CSV path.
    index : bool, default=False
        Whether to write the DataFrame index.
    compression : str or dict, optional
        Compression argument passed to ``pandas.DataFrame.to_csv``.
    """
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")

    dataframe.to_csv(tmp_path, index=index, compression=compression)
    tmp_path.replace(path)


def enforce_scenario_index(dataframe, scenario_index_lookup):
    """
    Replace scenario_index using the scenario metadata lookup.

    Result-summary DataFrames identify scenarios using ``scenario``.
    Trial-metadata DataFrames use ``scenario_file``. Both forms are
    supported.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame whose scenario index should be assigned.
    scenario_index_lookup : pandas.DataFrame
        Lookup containing ``scenario_file`` and ``scenario_index``.

    Returns
    -------
    pandas.DataFrame
        A copy of the input with scenario indices assigned from the
        lookup.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    
    required_columns = {"scenario_file", "scenario_index"}
    missing_columns = required_columns.difference(
        scenario_index_lookup.columns
    )

    if missing_columns:
        raise KeyError(
            "scenario_index_lookup is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    
    df = dataframe.copy()
    lookup = scenario_index_lookup[
        ["scenario_file", "scenario_index"]
    ].copy()
    lookup = lookup.drop_duplicates(
        subset="scenario_file",
        keep="last",
    )

    if "scenario_index" in df.columns:
        df = df.drop(columns="scenario_index")

    if "scenario_file" in df.columns:
        merge_column = "scenario_file"
    elif "scenario" in df.columns:
        merge_column = "scenario"
        lookup = lookup.rename(
            columns={"scenario_file": "scenario"}
        )
    else:
        raise KeyError(
            "Cannot assign scenario_index because the DataFrame "
            "contains neither 'scenario' nor 'scenario_file'. "
            f"Available columns are: {df.columns.to_list()}"
        )

    df = df.merge(
        lookup,
        on=merge_column,
        how="left",
        validate="many_to_one",
    )

    missing_scenarios = df.loc[
        df["scenario_index"].isna(),
        merge_column,
    ].unique()

    if len(missing_scenarios) > 0:
        raise ValueError(
            "These scenarios are not present in the scenario "
            f"index lookup: {missing_scenarios}"
        )

    df["scenario_index"] = (
        df["scenario_index"].astype(int)
    )

    return df


def upsert_trial_metadata(
    row, 
    metadata_path, 
    scenario_index_lookup,
):
    """Add or replace one trial row in the trial metadata CSV.

    ``scenario_file`` and ``trial`` form the unique trial identifier.
    This function should be called only by the main process, not by a
    multiprocessing worker.

    Parameters
    ----------
    row : mapping
        Metadata values for one completed trial.
    metadata_path : path-like
        Trial metadata CSV path.
    scenario_index_lookup : pandas.DataFrame
        Lookup used to enforce stable scenario indices.
    """
    metadata_path = Path(metadata_path)
    new_row = pd.DataFrame([row])

    if metadata_path.exists():
        existing = pd.read_csv(metadata_path)

        same_trial = (
            (existing["scenario_file"] == row["scenario_file"])
            & (existing["trial"] == row["trial"])
        )

        updated = pd.concat(
            [existing.loc[~same_trial], new_row],
            ignore_index=True,
        )
    else:
        updated = new_row

    updated = enforce_scenario_index(
        updated,
        scenario_index_lookup=scenario_index_lookup,
    )

    updated = (
        updated
        .sort_values(["scenario_index", "trial"])
        .reset_index(drop=True)
    )

    write_csv_safely(updated, metadata_path, index=False)


def run_one_des_trial_from_config(trial_config):
    """
    Run one DES trial for one hybrid scenario.

    This function is designed to be called by multiprocessing.Pool. It should
    only write files that are unique to this scenario/trial combination.

    Parameters
    ----------
    trial_config : dict
        Dictionary containing all parameters and file paths required to run
        one trial.

    Returns
    -------
    dict
        One metadata row describing the completed trial.
    """
    scenario_name = trial_config["scenario_name"]
    scenario_title = trial_config["scenario_title"]
    scenario_index = int(trial_config["scenario_index"])
    trial = int(trial_config["trial"])
    seed = int(trial_config["seed"])

    records_path = Path(trial_config["records_path"])
    patient_path = Path(trial_config["patient_path"])
    cohort_path = Path(trial_config["cohort_path"])
    activity_path = Path(trial_config["activity_path"])

    scenario_index_lookup = pd.DataFrame(
        trial_config["scenario_index_lookup_records"]
    )

    if records_path.exists():
        print(
            f"  Skipping DES run for trial {trial}; "
            "raw records already exist."
        )

        records_df = pd.read_csv(records_path)

    else:
        ciw.seed(seed)

        print(
            f"  Running DES trial {trial} for {scenario_name}; "
            f"seed={seed}"
        )

        gp_arrival_rates = des.make_gp_arrival_rates(
            lambdas=trial_config["lambdas_with_warmup"],
            t=trial_config["t_with_warmup"],
            max_sample_date=trial_config["des_run_time"],
        )

        other_referral_arrivals = des.make_other_referral_arrival_rates(
            weekday_rates=trial_config["weekday_rates"],
            endpoints=trial_config["endpoints"],
            max_sample_date=trial_config["des_run_time"],
        )

        subspecialty_service_dists = (
            des.make_deterministic_service_distributions(
                trial_config["service_values"]
            )
        )

        subspec_dict = {
            subspec: idx
            for idx, subspec in enumerate(trial_config["subspecs"])
        }

        jockeying_class = des.JockeyRouting(
            pdfa_matrix=trial_config["pdfas"],
            alphabet=trial_config["alphabets"],
            activity_dict=trial_config["activity_dict"],
            subspec_dict=subspec_dict,
            pre_op_letter=trial_config["pre_op_letter"],
            elective_surgery_letter=trial_config["elective_surgery_letter"],
        )

        reneging_distribution = des.PreOpExpiryDist(
            activity_dict=trial_config["activity_dict"],
            subspec_dict=subspec_dict,
            pre_op_letter=trial_config["pre_op_letter"],
            elective_surgery_letter=trial_config["elective_surgery_letter"],
        )

        network = des.get_network(
            alphabets=trial_config["alphabets"],
            subspecialties=trial_config["subspecs"],
            subspecialty_service_dists=subspecialty_service_dists,
            emergency_nodes=trial_config["emergency_nodes"],
            subspecialty_class=jockeying_class,
            reneging_distribution=reneging_distribution,
            subspec_probs_low=trial_config["subspec_probs_low"],
            subspec_probs_medium=trial_config["subspec_probs_medium"],
            subspec_probs_high=trial_config["subspec_probs_high"],
            gp_arrival_rates=gp_arrival_rates,
            other_arrival_rates=other_referral_arrivals,
        )

        records_df = des.run_des_trial(
            network=network,
            run_time=trial_config["des_run_time"],
            progress_bar=False,
        )

        write_csv_safely(
            records_df,
            records_path,
            index=False,
        )

    patient_records = res.remove_warmup_patients(
        records_df=records_df,
        warmup_days=trial_config["warm_up"],
        reset_time=True,
    )

    activity_records = res.remove_warmup_activity_records(
        records_df=records_df,
        warmup_days=trial_config["warm_up"],
        reset_time=True,
    )

    if (
        patient_path.exists()
        and cohort_path.exists()
        and activity_path.exists()
    ):
        print(
            f"  Skipping summarisation for trial {trial}; "
            "summary files already exist."
        )

        patient_df = pd.read_csv(patient_path)

    else:
        patient_df, cohort_df, activity_summary_df = (
            res.summarise_des_records(
                patient_records=patient_records,
                activity_records=activity_records,
                subspecialties=trial_config["subspecs"],
                activity_dictionary=trial_config["activity_dict"],
                scenario_name=scenario_name,
                scenario_index=scenario_index,
                trial=trial,
                seed=seed,
                nodes=trial_config["nodes"],
            )
        )

        patient_df = enforce_scenario_index(
            patient_df,
            scenario_index_lookup=scenario_index_lookup,
        )

        cohort_df = enforce_scenario_index(
            cohort_df,
            scenario_index_lookup=scenario_index_lookup,
        )

        activity_summary_df = enforce_scenario_index(
            activity_summary_df,
            scenario_index_lookup=scenario_index_lookup,
        )

        write_csv_safely(
            patient_df,
            patient_path,
            index=False,
        )

        write_csv_safely(
            cohort_df,
            cohort_path,
            index=False,
        )

        write_csv_safely(
            activity_summary_df,
            activity_path,
            index=False,
        )

        print(f"  Saved summary files for trial {trial} of scenario {scenario_name}")

    trial_row = {
        "scenario_index": scenario_index,
        "policy": scenario_title,
        "scenario_file": scenario_name,
        "trial": trial,
        "seed": seed,
        "records_file": trial_config["records_file_relative"],
        "patient_summary_file": trial_config["patient_file_relative"],
        "cohort_summary_file": trial_config["cohort_file_relative"],
        "activity_summary_file": trial_config["activity_file_relative"],
        "n_records_raw": len(records_df),
        "n_records_post_warmup": len(activity_records),
        "n_patients": (
            patient_df["id_number"].nunique()
            if "id_number" in patient_df.columns
            else len(patient_df)
        ),
        "warmup_days": trial_config["warm_up"],
        "analysis_run_time": trial_config["run_time"],
        "des_run_time": trial_config["des_run_time"],
    }

    print(
        f"  Finished DES trial {trial} for {scenario_name}; "
        f"n_records={len(records_df)}"
    )

    return trial_row