"""
Prepare and summarise discrete-event simulation results.

This module maps raw Ciw records to model activities and
subspecialties, removes simulation warm-up periods, and creates
patient-, cohort-, and activity-level summaries.
"""

import numpy as np
import pandas as pd

SEVERITY_CATEGORIES = ["Low", "Medium", "High"]


def first_non_null(series):
    """
    Return the first non-missing value in a Series.

    This function is used in group aggregations to obtain one value per
    patient for fields such as subspecialty and severity.

    Parameters
    ----------
    series : pandas.Series
        Values from which to select the first non-missing item.

    Returns
    -------
    object
        First non-missing value, or ``numpy.nan`` when every value is
        missing.
    """
    non_missing = series.dropna()

    if len(non_missing) == 0:
        return np.nan
    
    return non_missing.iloc[0]


def get_node_lookup(nodes, subspecialties, activity_dict):
    """
    Map Ciw node numbers to subspecialty and activity labels.

    The first two nodes are treated as dummy entry nodes. All remaining
    nodes are interpreted as repeated activity blocks, with one block
    for each subspecialty.

    Parameters
    ----------
    nodes : list
        Node labels in Ciw node-number order.
    subspecialties : list
        Subspecialties in activity-block order.
    activity_dict : dict
        Mapping from activity letters to node indices.

    Returns
    -------
    pandas.DataFrame
        Node numbers and their corresponding activity, subspecialty,
        and entry-node indicators.

    Raises
    ------
    ValueError
        If ``activity_dict`` contains no activities.
    """
    ordered_activities = [
        letter for letter, _ in sorted(
            activity_dict.items(), 
            key=lambda x: x[1],
        )
    ]
    n_activities = len(ordered_activities)

    rows = []
    
    for node_id in range(1, len(nodes) + 1):
        node_label = nodes[node_id - 1]

        if node_id <= 2 or node_label == "*":
            rows.append(
                {
                    "node": node_id,
                    "dummy_node": True,
                    "node_label": node_label,
                    "subspecialty_from_node": np.nan,
                    "activity_letter": np.nan,
                }
            )
        else:
            offset = node_id - 3
            subspec_idx = offset // n_activities
            activity_idx = offset % n_activities

            rows.append(
                {
                    "node": node_id,
                    "dummy_node": False,
                    "node_label": node_label,
                    "subspecialty_from_node": (
                        subspecialties[subspec_idx]
                    ),
                    "activity_letter": (
                        ordered_activities[activity_idx]
                    ),
                }
            )

    return pd.DataFrame(rows)


def remove_warmup_patients(
    records_df, 
    warmup_days, 
    reset_time=True,
):
    """
    Remove patients whose first arrival occurs during warm-up.

    All records belonging to a patient are removed when that patient's
    first observed arrival is before ``warmup_days``.

    Parameters
    ----------
    records_df : pandas.DataFrame
        Raw simulation records.
    warmup_days : int or float
        Length of the warm-up period.
    reset_time : bool, default=True
        Whether to subtract ``warmup_days`` from the remaining
        simulation time columns.

    Returns
    -------
    pandas.DataFrame
        Records for patients arriving after the warm-up period.
    """
    records_df = records_df.copy()

    first_arrivals = (
        records_df
        .groupby("id_number")["arrival_date"]
        .min()
    )
    keep_ids = first_arrivals[
        first_arrivals >= warmup_days
    ].index

    post_warmup_records = records_df[
        records_df["id_number"].isin(keep_ids)
    ].copy()

    if reset_time:
        time_columns = [
            "arrival_date",
            "service_start_date",
            "service_end_date",
            "exit_date",
        ]

        for col in time_columns:
            if col in post_warmup_records.columns:
                post_warmup_records[col] = (
                    post_warmup_records[col] - warmup_days
                )

    return post_warmup_records


def remove_warmup_activity_records(
    records_df, 
    warmup_days, 
    reset_time=True,
):
    """"
    Remove activity records that occur during the warm-up period.

    Unlike :func:`remove_warmup_patients`, this function removes
    individual records rather than removing every record belonging to a
    patient who entered during warm-up.

    Parameters
    ----------
    records_df : pandas.DataFrame
        Raw simulation records.
    warmup_days : int or float
        Length of the warm-up period.
    reset_time : bool, default=True
        Whether to subtract ``warmup_days`` from the remaining
        simulation time columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with warm-up activity records removed.
    """
    records_df = records_df.copy()

    post_warmup_records = records_df[
        records_df["arrival_date"] >= warmup_days
    ].copy()

    if reset_time:
        time_columns = [
            "arrival_date",
            "service_start_date",
            "service_end_date",
            "exit_date",
        ]

        for col in time_columns:
            if col in post_warmup_records.columns:
                post_warmup_records[col] = (
                    post_warmup_records[col] - warmup_days
                )

    return post_warmup_records


def prepare_results_df(
    raw_df,
    nodes,
    subspecialties,
    activity_dict,
    keep_entry_nodes=False,
):
    """
    Prepare raw Ciw records for DES results analysis.

    Processing includes:

    - mapping node numbers to activities and subspecialties;
    - creating an ordered severity column;
    - filling missing subspecialties from customer classes;
    - identifying records that leave the system;
    - calculating total observed time at each node;
    - identifying records with positive waiting times; and
    - optionally removing entry-node records.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Raw Ciw records.
    nodes : list
        Node labels in Ciw node-number order.
    subspecialties : list
        Subspecialties in activity-block order.
    activity_dict : dict
        Mapping from activity letters to node indices.
    keep_entry_nodes : bool, default=False
        Whether to retain the two model-entry nodes.

    Returns
    -------
    pandas.DataFrame
        Processed activity-level records.
    """
    results_df = raw_df.copy()

    node_lookup = get_node_lookup(
        nodes, 
        subspecialties, 
        activity_dict,
    )
    results_df = results_df.merge(
        node_lookup, 
        on="node", 
        how="left",
    )

    results_df["severity"] = pd.Categorical(
        results_df["level"],
        categories=SEVERITY_CATEGORIES,
        ordered=True,
    )

    customer_subspecialty = results_df[
        "customer_class"
    ].where(
        results_df["customer_class"].isin(subspecialties)
    )

    results_df["subspecialty"] = results_df[
        "subspecialty_from_node"
    ].fillna(customer_subspecialty)

    results_df["left_system"] = results_df[
        "destination"
    ].eq(-1)

    results_df["total_node_time"] = (
        results_df["waiting_time"].fillna(0)
        + results_df["service_time"].fillna(0)
        + results_df["time_blocked"].fillna(0)
    )

    results_df["had_to_wait"] = results_df["waiting_time"].fillna(0) > 0

    if not keep_entry_nodes:
        results_df = results_df.loc[
            ~results_df["dummy_node"]
        ].copy()

    return results_df.reset_index(drop=True)


def get_patient_summary(records_dataframe):
    """
    Create patient-level summaries from activity-level records.

    Each row represents one patient and includes their pathway,
    completion status, waiting times, service times, and number of
    completed activities.

    Parameters
    ----------
    records_dataframe : pandas.DataFrame
        Processed activity-level records returned by
        :func:`prepare_results_df`.

    Returns
    -------
    pandas.DataFrame
        Patient-level summary records.
    """
    ordered = records_dataframe.sort_values(
        [
            "id_number", 
            "arrival_date", 
            "service_start_date", 
            "service_end_date",
        ]
    ).copy()

    ordered["completed_activity_letter"] = ordered[
        "activity_letter"
    ].where(
        ordered["record_type"].eq("service")
    )

    patient_df = (
        ordered.groupby("id_number", dropna=False)
        .agg(
            subspecialty=("subspecialty", first_non_null),
            severity=("severity", first_non_null),
            referral_source=("referral_source", first_non_null),
            arrival_date=("arrival_date", "min"),
            exit_date=("exit_date", "max"),
            n_records=("node", "size"),
            n_completed_activities=(
                "record_type", 
                lambda x: x.eq("service").sum()
            ),
            ever_reneged=(
                "record_type", 
                lambda x: x.eq("renege").any()
            ),
            total_reneges=(
                "record_type", 
                lambda x: x.eq("renege").sum()
            ),
            total_waiting_time=("waiting_time", "sum"),
            total_service_time=("service_time", "sum"),
            mean_waiting_time=("waiting_time", "mean"),
            median_waiting_time=("waiting_time", "median"),
            max_waiting_time=("waiting_time", "max"),
            ever_waited=("had_to_wait", "any"),
            n_waited_activities=("had_to_wait", "sum"),
            pathway=(
                "activity_letter",
                lambda x: "".join(x.dropna().astype(str)),
            ),
            completed_pathway=(
                "completed_activity_letter",
                lambda x: "".join(x.dropna().astype(str)),
            ),
        )
        .reset_index()
    )

    last_destination = ordered.groupby(
        "id_number"
    )["destination"].last()

    patient_df["completed"] = (
        patient_df["id_number"].map(last_destination).eq(-1)
    )

    patient_df["time_in_system"] = (
        patient_df["exit_date"] - patient_df["arrival_date"]
    )

    patient_df["severity"] = pd.Categorical(
        patient_df["severity"],
        categories=SEVERITY_CATEGORIES,
        ordered=True,
    )

    return patient_df


def get_cohort_summary(patient_df):
    """
    Create cohort-level summaries from patient-level DES results.

    Summaries are generated for combinations of subspecialty, severity,
    and referral source, as well as for broader aggregation levels.

    Measures affected by incomplete pathways are reported for both all
    observed patients and completed patients only.

    Parameters
    ----------
    patient_df : pandas.DataFrame
        Patient-level results returned by
        :func:`get_patient_summary`.

    Returns
    -------
    pandas.DataFrame
        Cohort-level measures at several aggregation levels.
    """
    cohort_source = patient_df.copy()

    cohort_source["completed"] = (
        cohort_source["completed"]
        .fillna(False)
        .astype(bool)
    )
    completed_mask = cohort_source["completed"]

    cohort_source["completed_time_in_system"] = cohort_source[
        "time_in_system"
    ].where(completed_mask)

    cohort_source["completed_total_waiting_time"] = cohort_source[
        "total_waiting_time"
    ].where(completed_mask)

    cohort_source["completed_total_service_time"] = cohort_source[
        "total_service_time"
    ].where(completed_mask)

    cohort_source["completed_n_completed_activities"] = cohort_source[
        "n_completed_activities"
    ].where(completed_mask)

    dimensions = [
        "subspecialty",
        "severity",
        "referral_source",
    ]

    aggregations = {
        "n_patients": (
            "id_number",
            "nunique",
        ),
        "n_completed": (
            "completed",
            "sum",
        ),
        "completion_rate": (
            "completed",
            "mean",
        ),
        "n_patients_reneged": (
            "ever_reneged",
            "sum",
        ),
        "reneging_rate": (
            "ever_reneged",
            "mean",
        ),
        "mean_total_reneges": (
            "total_reneges",
            "mean",
        ),
        "proportion_ever_waited": (
            "ever_waited",
            "mean",
        ),
        "mean_n_waited_activities": (
            "n_waited_activities",
            "mean",
        ),
        "mean_time_in_system_all_patients": (
            "time_in_system",
            "mean",
        ),
        "median_time_in_system_all_patients": (
            "time_in_system",
            "median",
        ),
        "p90_time_in_system_all_patients": (
            "time_in_system",
            lambda x: x.quantile(0.90),
        ),
        "mean_time_in_system_completed": (
            "completed_time_in_system",
            "mean",
        ),
        "median_time_in_system_completed": (
            "completed_time_in_system",
            "median",
        ),
        "p90_time_in_system_completed": (
            "completed_time_in_system",
            lambda x: x.quantile(0.90),
        ),
        "mean_total_wait_all_patients": (
            "total_waiting_time",
            "mean",
        ),
        "median_total_wait_all_patients": (
            "total_waiting_time",
            "median",
        ),
        "p90_total_wait_all_patients": (
            "total_waiting_time",
            lambda x: x.quantile(0.90),
        ),
        "mean_total_wait_completed": (
            "completed_total_waiting_time",
            "mean",
        ),
        "median_total_wait_completed": (
            "completed_total_waiting_time",
            "median",
        ),
        "p90_total_wait_completed": (
            "completed_total_waiting_time",
            lambda x: x.quantile(0.90),
        ),
        "mean_total_service_all_patients": (
            "total_service_time",
            "mean",
        ),
        "mean_total_service_completed": (
            "completed_total_service_time",
            "mean",
        ),
        "mean_n_completed_activities_all_patients": (
            "n_completed_activities",
            "mean",
        ),
        "median_n_completed_activities_all_patients": (
            "n_completed_activities",
            "median",
        ),
        "mean_n_completed_activities_completed": (
            "completed_n_completed_activities",
            "mean",
        ),
        "median_n_completed_activities_completed": (
            "completed_n_completed_activities",
            "median",
        ),
    }

    def summarise(group_columns, aggregation_level):
        """Calculate one aggregation level."""
        if group_columns:
            summary = (
                cohort_source.groupby(
                    group_columns,
                    dropna=False,
                    observed=False,
                )
                .agg(**aggregations)
                .reset_index()
            )
        else:
            summary = (
                cohort_source.assign(_overall_group="All")
                .groupby("_overall_group")
                .agg(**aggregations)
                .reset_index(drop=True)
            )

        for column in dimensions:
            if column not in summary.columns:
                summary[column] = "All"
            else:
                summary[column] = summary[column].astype(object)

        summary["aggregation_level"] = aggregation_level

        return summary

    summaries = [
        summarise(
            [
                "subspecialty", 
                "severity",
                "referral_source",
            ],
            "Subspecialty x severity x referral source",
        ),
        summarise(
            [
                "subspecialty", 
                "severity",
            ],
            "Subspecialty x severity",
        ),
        summarise(
            [
                "subspecialty", 
                "referral_source",
            ],
            "Subspecialty x referral source",
        ),
        summarise(
            [
                "severity", 
                "referral_source",
            ],
            "Severity x referral source",
        ),
        summarise(
            ["subspecialty"],
            "Subspecialty",
        ),
        summarise(
            ["severity"],
            "Severity",
        ),
        summarise(
            ["referral_source"],
            "Referral source",
        ),
        summarise(
            [],
            "Overall",
        ),
    ]

    cohort_df = pd.concat(
        summaries,
        ignore_index=True,
    )

    cohort_df["n_incomplete"] = (
        cohort_df["n_patients"]
        - cohort_df["n_completed"]
    )

    column_order = [
        "aggregation_level",
        "subspecialty",
        "severity",
        "referral_source",
        *aggregations.keys(),
        "n_incomplete",
    ]

    return cohort_df[column_order]


def get_activity_summary(activity_df):
    """
    Create activity-level summaries from processed DES records.

    Each row represents one subspecialty, severity, and activity
    combination. Measures include waiting times, service times, queue
    sizes, and reneging counts.

    Parameters
    ----------
    activity_df : pandas.DataFrame
        Processed activity-level records returned by
        :func:`prepare_results_df`.

    Returns
    -------
    pandas.DataFrame
        Activity-level summary records.
    """
    activity_source = activity_df.copy()

    activity_source["is_service"] = activity_source[
        "record_type"
    ].eq("service")
    activity_source["is_renege"] = activity_source[
        "record_type"
    ].eq("renege")
    activity_source["is_wait"] = activity_source[
        "waiting_time"
    ].fillna(0).gt(0)

    activity_source[
        "completed_service_time"
    ] = activity_source["service_time"].where(
        activity_source["is_service"]
    )

    activity_source[
        "renege_waiting_time"
    ] = activity_source["waiting_time"].where(
        activity_source["is_renege"]
    )

    activity_summary = (
        activity_source.groupby(
            [
                "subspecialty", 
                "severity", 
                "activity_letter",
            ],
            dropna=False,
            observed=False,
        )
        .agg(
            n_records=("node", "size"),
            n_patients=("id_number", "nunique"),
            n_completed_activities=("is_service", "sum"),
            n_reneges=("is_renege", "sum"),
            n_waited_records=("is_wait", "sum"),
            proportion_waited_records=("is_wait", "mean"),
            mean_waiting_time=("waiting_time", "mean"),
            median_waiting_time=("waiting_time", "median"),
            p90_waiting_time=(
                "waiting_time", 
                lambda x: x.quantile(0.90)
            ),
            mean_wait_to_renege=("renege_waiting_time", "mean"),
            median_wait_to_renege=("renege_waiting_time", "median"),
            p90_wait_to_renege=(
                "renege_waiting_time",
                lambda x: x.quantile(0.90),
            ),
            total_service_time=("completed_service_time", "sum"),
            mean_service_time=("completed_service_time", "mean"),
            median_service_time=(
                "completed_service_time",
                "median",
            ),
            mean_queue_at_arrival=(
                "queue_size_at_arrival",
                "mean",
            ),
            median_queue_at_arrival=(
                "queue_size_at_arrival",
                "median",
            ),
            p90_queue_at_arrival=(
                "queue_size_at_arrival",
                lambda x: x.quantile(0.90),
            ),
            mean_queue_at_departure=(
                "queue_size_at_departure",
                "mean",
            ),
            median_queue_at_departure=(
                "queue_size_at_departure",
                "median",
            ),
            p90_queue_at_departure=(
                "queue_size_at_departure",
                lambda x: x.quantile(0.90),
            ),
        )
        .reset_index()
        .sort_values(["subspecialty", "severity", "activity_letter"])
        .reset_index(drop=True)
    )

    return activity_summary


def add_scenario_trial_columns(
    df, 
    scenario_name, 
    scenario_index, 
    trial, 
    seed,
):
    """
    Add scenario and replication identifiers to a result DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Results to label.
    scenario_name : str
        Scenario name.
    scenario_index : int
        Scenario index.
    trial : int
        Trial or replication number.
    seed : int
        Random seed used for the trial.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with four identifying columns added.
    """
    labelled_df = df.copy()
    labelled_df.insert(0, "seed", seed)
    labelled_df.insert(0, "trial", trial)
    labelled_df.insert(0, "scenario_index", scenario_index)
    labelled_df.insert(0, "scenario", scenario_name)
    return labelled_df


def summarise_des_records(
        patient_records, 
        activity_records, 
        subspecialties, 
        activity_dictionary, 
        scenario_name, 
        scenario_index, 
        trial, 
        seed, 
        nodes):
    """
    Convert raw DES records into three summary DataFrames.

    Parameters
    ----------
    patient_records : pandas.DataFrame
        Raw records used to calculate patient-level results.
    activity_records : pandas.DataFrame
        Raw records used to calculate activity-level results.
    subspecialties : list
        Subspecialties in activity-block order.
    activity_dictionary : dict
        Mapping from activity letters to node indices.
    scenario_name : str
        Scenario name used to label the results.
    scenario_index : int
        Scenario index used to label the results.
    trial : int
        Trial or replication number.
    seed : int
        Random seed used for the trial.
    nodes : list
        Node labels in Ciw node-number order.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Patient-level, cohort-level, and activity-level summaries.
    """
    patient_activity_df = prepare_results_df(
        raw_df=patient_records,
        nodes=nodes,
        subspecialties=subspecialties,
        activity_dict=activity_dictionary,
        keep_entry_nodes=False,
    )

    activity_df = prepare_results_df(
        raw_df=activity_records,
        nodes=nodes,
        subspecialties=subspecialties,
        activity_dict=activity_dictionary,
        keep_entry_nodes=False,
    )

    patient_df = get_patient_summary(patient_activity_df)
    cohort_df = get_cohort_summary(patient_df)
    activity_summary_df = get_activity_summary(activity_df)

    patient_df = add_scenario_trial_columns(
        df=patient_df,
        scenario_name=scenario_name,
        scenario_index=scenario_index,
        trial=trial,
        seed=seed,
    )
    cohort_df = add_scenario_trial_columns(
        df=cohort_df,
        scenario_name=scenario_name,
        scenario_index=scenario_index,
        trial=trial,
        seed=seed,
    )
    activity_summary_df = add_scenario_trial_columns(
        df=activity_summary_df,
        scenario_name=scenario_name,
        scenario_index=scenario_index,
        trial=trial,
        seed=seed,
    )

    return patient_df, cohort_df, activity_summary_df
