"""
Run either System Dynamics deterioration experiment.

The stock proportional and boundary shift formulations have separate configuration
modules but share this runner. Each experiment writes to its own output
files, allowing the experiments to be run independently or together.

Examples
--------
Run the stock proportional experiment:

``python run_sd_deterioration.py stock_proportional``

Run the boundary shift experiment:

``python run_sd_deterioration.py boundary_shift``

Run both experiments:

``python run_sd_deterioration.py all``
"""

import argparse
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from experiment_paths import (
    EXP_DIR,
    SRC_DIR,
    RAW_DIR,
)


# Make the reusable project modules importable when this file is run
# directly.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Make the experiment configuration package importable when this file
# is run directly.
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from hybridsim import sd_component as sd

from configs import common


EXPERIMENT_CONFIG_MODULES = {
    "stock_proportional": (
        "configs.stock_proportional_deterioration"
    ),
    "boundary_shift": (
        "configs.boundary_shift_deterioration"
    ),
}

METADATA_COLUMNS = [
    "scenario_index",
    "scenario",
    "deterioration_model",
    "deterioration_label",
    "stock_proportional_mean_time_days",
    "boundary_shift_proportion",
    "gatekeeping_policy",
    "description",
]


def parse_arguments():
    """
    Parse the requested deterioration experiment.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the stock proportional or boundary shift SD "
            "deterioration experiment."
        )
    )

    parser.add_argument(
        "experiment",
        choices=[
            *EXPERIMENT_CONFIG_MODULES,
            "all",
        ],
        help=(
            "Experiment to run. Use 'all' to run both "
            "experiments sequentially."
        ),
    )

    return parser.parse_args()


def get_selected_configurations(experiment):
    """
    Load the scenario configuration modules requested by the user.

    Parameters
    ----------
    experiment : str
        ``stock_proportional``, ``boundary_shift``, or ``all``.

    Returns
    -------
    list of module
        Imported scenario configuration modules.
    """
    if experiment == "all":
        selected_names = list(
            EXPERIMENT_CONFIG_MODULES
        )
    else:
        selected_names = [experiment]

    return [
        importlib.import_module(
            EXPERIMENT_CONFIG_MODULES[name]
        )
        for name in selected_names
    ]


def get_output_paths(experiment_name):
    """
    Return output paths for one deterioration experiment.

    Parameters
    ----------
    experiment_name : str
        Stable name used in the output filenames.

    Returns
    -------
    dict
        Paths for arrays, scenario metadata, and optional time series.
    """
    return {
        "arrays_path": (
            RAW_DIR / f"sd_arrays_{experiment_name}.npz"
        ),
        "scenario_metadata_path": (
            RAW_DIR
            / f"scenario_metadata_{experiment_name}.csv"
        ),
        "timeseries_path": (
            RAW_DIR
            / f"sd_timeseries_{experiment_name}.csv"
        ),
    }


def write_csv_safely(
    dataframe,
    path,
    index=False,
):
    """
    Write a CSV through a temporary file before replacing the target.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Data to write.
    path : path-like
        Destination CSV path.
    index : bool, default=False
        Whether to write the DataFrame index.
    """
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")

    dataframe.to_csv(tmp_path, index=index)
    tmp_path.replace(path)


def write_npz_safely(
    path,
    **arrays,
):
    """
    Write an NPZ file through a temporary file before replacing the target.

    Parameters
    ----------
    path : path-like
        Destination NPZ path.
    **arrays
        Named arrays passed to ``numpy.savez``.
    """
    path = Path(path)
    tmp_path = path.with_name(
        f"{path.stem}.tmp{path.suffix}"
    )

    np.savez(tmp_path, **arrays)
    tmp_path.replace(path)


def prepare_scenarios(scenario_definitions):
    """
    Convert configured scenario definitions into a DataFrame.

    Parameters
    ----------
    scenario_definitions : list of dict
        Configured scenario definitions.

    Returns
    -------
    pandas.DataFrame
        Validated scenario definitions.
    """
    scenarios = pd.DataFrame(scenario_definitions)

    optional_metadata_defaults = {
        "stock_proportional_mean_time_days": np.nan,
        "boundary_shift_proportion": np.nan,
    }

    for column, default_value in optional_metadata_defaults.items():
        if column not in scenarios.columns:
            scenarios[column] = default_value

    required_columns = set(
        METADATA_COLUMNS[1:]
        + [
            "deterioration_function",
            "gatekeeping_function",
        ]
    )
    missing_columns = required_columns.difference(
        scenarios.columns
    )

    if missing_columns:
        raise ValueError(
            "Scenario definitions are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if scenarios.empty:
        raise ValueError(
            "At least one deterioration scenario must be configured."
        )

    duplicate_names = scenarios.loc[
        scenarios["scenario"].duplicated(keep=False),
        "scenario",
    ].unique()

    if len(duplicate_names) > 0:
        raise ValueError(
            "Scenario names must be unique. Duplicate names: "
            f"{duplicate_names}"
        )

    scenarios.insert(
        0,
        "scenario_index",
        np.arange(len(scenarios)),
    )

    return scenarios


def validate_run_configuration():
    """
    Validate the shared simulation controls and SD parameters.
    """
    if common.RUN_TIME <= 0:
        raise ValueError(
            "RUN_TIME must be greater than zero."
        )

    if common.TIME_POINTS < 2:
        raise ValueError(
            "TIME_POINTS must be at least 2."
        )

    if (
        common.SD_PARAMS["time_points"]
        != common.TIME_POINTS
    ):
        raise ValueError(
            "SD_PARAMS['time_points'] must equal "
            "TIME_POINTS."
        )

    splits = np.asarray(
        common.SD_PARAMS["unwell_splits"],
        dtype=float,
    )

    if splits.shape != (
        len(common.SEVERITY_LEVELS),
    ):
        raise ValueError(
            "UNWELL_SPLITS must contain one value "
            "per severity level."
        )

    if (
        np.any(splits < 0)
        or not np.isclose(splits.sum(), 1.0)
    ):
        raise ValueError(
            "UNWELL_SPLITS must be non-negative "
            "and sum to 1."
        )

    if common.SD_PARAMS["initial_population"] <= 0:
        raise ValueError(
            "initial_population must be greater than zero."
        )

    if not (
        0
        <= common.SD_PARAMS["unwell_proportion"]
        <= 1
    ):
        raise ValueError(
            "unwell_proportion must lie between 0 and 1."
        )

    if not (
        0
        <= common.SD_PARAMS["presenting_proportion"]
        <= 1
    ):
        raise ValueError(
            "presenting_proportion must lie between "
            "0 and 1."
        )

    if common.SD_PARAMS["referral_threshold"] < 0:
        raise ValueError(
            "referral_threshold cannot be negative."
        )

    if any(
        rate < 0
        for rate in common.SD_PARAMS[
            "incidence_rates"
        ]
    ):
        raise ValueError(
            "incidence_rates cannot contain "
            "negative values."
        )

    if any(
        rate < 0
        for rate in common.SD_PARAMS[
            "recovery_rates"
        ]
    ):
        raise ValueError(
            "recovery_rates cannot contain "
            "negative values."
        )


def run_sd_scenario(
    deterioration_function,
    gatekeeping_function,
):
    """
    Run the SD component for one deterioration scenario.

    Parameters
    ----------
    deterioration_function : function
        Deterioration function supplied to the SD model.
    gatekeeping_function : function
        Gatekeeping function supplied to the SD model.

    Returns
    -------
    dict
        Time points, stock trajectories, and referral trajectories.
    """
    model = sd.SD(
        population_function=(
            sd.get_time_dependent_population_size(
                [
                    common.SD_PARAMS[
                        "initial_population"
                    ]
                ],
                common.RUN_TIME,
            )
        ),
        initial_unwell_proportion=(
            common.SD_PARAMS[
                "unwell_proportion"
            ]
        ),
        unwell_splits=common.SD_PARAMS[
            "unwell_splits"
        ],
        gatekeeping_function=gatekeeping_function,
        presenting_proportion=(
            common.SD_PARAMS[
                "presenting_proportion"
            ]
        ),
        deterioration_function=(
            deterioration_function
        ),
        incidence_function=(
            sd.get_time_dependent_incidence_rate(
                common.SD_PARAMS[
                    "incidence_rates"
                ],
                common.RUN_TIME,
            )
        ),
        recovery_function=(
            sd.get_time_dependent_recovery_rate(
                common.SD_PARAMS[
                    "recovery_rates"
                ],
                common.RUN_TIME,
            )
        ),
    )

    t = np.linspace(
        0,
        common.RUN_TIME,
        common.TIME_POINTS,
    )
    model.solve(t=t)

    stocks = np.asarray(model.P).T
    lambdas = np.asarray(model.lambdas).T

    expected_shape = (
        len(t),
        len(common.SEVERITY_LEVELS),
    )

    if stocks.shape != expected_shape:
        raise ValueError(
            "Unexpected stock array shape. "
            f"Expected {expected_shape}, "
            f"received {stocks.shape}."
        )

    if lambdas.shape != expected_shape:
        raise ValueError(
            "Unexpected lambda array shape. "
            f"Expected {expected_shape}, "
            f"received {lambdas.shape}."
        )

    return {
        "t": t,
        "stocks": stocks,
        "lambdas": lambdas,
    }


def result_to_timeseries_dataframe(
    scenario_row,
    result,
):
    """
    Convert one scenario result into a long-format time-series table.

    Parameters
    ----------
    scenario_row : pandas.Series
        Scenario definition and metadata.
    result : dict
        Output from ``run_sd_scenario``.

    Returns
    -------
    pandas.DataFrame
        One row per simulation time point.
    """
    stocks = result["stocks"]
    lambdas = result["lambdas"]

    dataframe = pd.DataFrame(
        {
            "scenario_index": int(
                scenario_row["scenario_index"]
            ),
            "scenario": scenario_row[
                "scenario"
            ],
            "deterioration_model": (
                scenario_row[
                    "deterioration_model"
                ]
            ),
            "deterioration_label": (
                scenario_row[
                    "deterioration_label"
                ]
            ),
            "stock_proportional_mean_time_days": (
                scenario_row[
                    "stock_proportional_mean_time_days"
                ]
            ),
            "boundary_shift_proportion": (
                scenario_row[
                    "boundary_shift_proportion"
                ]
            ),
            "gatekeeping_policy": (
                scenario_row[
                    "gatekeeping_policy"
                ]
            ),
            "description": scenario_row[
                "description"
            ],
            "t": result["t"],
            "stock_low": stocks[:, 0],
            "stock_medium": stocks[:, 1],
            "stock_high": stocks[:, 2],
            "lambda_low": lambdas[:, 0],
            "lambda_medium": lambdas[:, 1],
            "lambda_high": lambdas[:, 2],
        }
    )

    dataframe["stock_total"] = dataframe[
        [
            "stock_low",
            "stock_medium",
            "stock_high",
        ]
    ].sum(axis=1)

    dataframe["lambda_total"] = dataframe[
        [
            "lambda_low",
            "lambda_medium",
            "lambda_high",
        ]
    ].sum(axis=1)

    return dataframe


def print_initial_state():
    """
    Print the initial demand and referral values.
    """
    initial_unwell = (
        common.SD_PARAMS["initial_population"]
        * common.SD_PARAMS["unwell_proportion"]
    )
    initial_presenting = (
        initial_unwell
        * common.SD_PARAMS[
            "presenting_proportion"
        ]
    )
    initial_referrals = (
        initial_presenting
        * common.SD_PARAMS[
            "referral_threshold"
        ]
    )

    print(
        "Initial unwell population: "
        f"{initial_unwell:.2f}"
    )
    print(
        "Initial presenting population: "
        f"{initial_presenting:.2f}"
    )
    print(
        "Initial total referrals: "
        f"{initial_referrals:.4f}"
    )


def run_all_scenarios(scenarios):
    """
    Run all scenarios in one selected experiment.

    Parameters
    ----------
    scenarios : pandas.DataFrame
        Scenario definitions returned by
        ``prepare_scenarios``.

    Returns
    -------
    tuple
        Common time vector, stock array, lambda array,
        and optional long-format time-series table.
    """
    stock_outputs = []
    lambda_outputs = []
    timeseries_outputs = []
    common_t = None

    for _, scenario_row in scenarios.iterrows():
        scenario_name = scenario_row["scenario"]

        print()
        print(f"Running scenario: {scenario_name}")

        result = run_sd_scenario(
            deterioration_function=(
                scenario_row[
                    "deterioration_function"
                ]
            ),
            gatekeeping_function=(
                scenario_row[
                    "gatekeeping_function"
                ]
            ),
        )

        scenario_t = np.asarray(result["t"])

        if common_t is None:
            common_t = scenario_t
        elif not np.array_equal(
            common_t,
            scenario_t,
        ):
            raise ValueError(
                "All scenarios must use the same "
                "time grid."
            )

        stock_outputs.append(
            result["stocks"]
        )
        lambda_outputs.append(
            result["lambdas"]
        )

        if common.WRITE_TIMESERIES_CSV:
            timeseries_outputs.append(
                result_to_timeseries_dataframe(
                    scenario_row=scenario_row,
                    result=result,
                )
            )

    stocks = np.stack(
        stock_outputs,
        axis=0,
    )
    lambdas = np.stack(
        lambda_outputs,
        axis=0,
    )

    if common.WRITE_TIMESERIES_CSV:
        timeseries = pd.concat(
            timeseries_outputs,
            ignore_index=True,
        )
    else:
        timeseries = None

    return (
        common_t,
        stocks,
        lambdas,
        timeseries,
    )


def run_experiment(scenario_config):
    """
    Run one selected deterioration experiment.

    Parameters
    ----------
    scenario_config : module
        Stock proportional or boundary shift scenario
        configuration module.
    """
    experiment_name = (
        scenario_config.EXPERIMENT_NAME
    )
    scenarios = prepare_scenarios(
        scenario_config.SCENARIO_DEFINITIONS
    )
    output_paths = get_output_paths(
        experiment_name
    )

    print()
    print(f"Experiment: {experiment_name}")
    print(
        f"Running {len(scenarios)} SD "
        "deterioration scenario(s)."
    )

    (
        common_t,
        stocks,
        lambdas,
        timeseries,
    ) = run_all_scenarios(scenarios)

    scenario_metadata = scenarios[
        METADATA_COLUMNS
    ].copy()

    write_npz_safely(
        output_paths["arrays_path"],
        t=common_t,
        stocks=stocks,
        lambdas=lambdas,
        severity_levels=np.asarray(
            common.SEVERITY_LEVELS
        ),
        scenario_names=scenario_metadata[
            "scenario"
        ].to_numpy(dtype=str),
        scenario_indices=scenario_metadata[
            "scenario_index"
        ].to_numpy(dtype=int),
        experiment_name=np.asarray(
            experiment_name
        ),
    )

    write_csv_safely(
        scenario_metadata,
        output_paths[
            "scenario_metadata_path"
        ],
        index=False,
    )

    if timeseries is not None:
        write_csv_safely(
            timeseries,
            output_paths["timeseries_path"],
            index=False,
        )

    print()
    print(
        "Saved SD arrays to: "
        f"{output_paths['arrays_path']}"
    )
    print(
        "Saved scenario metadata to: "
        f"{output_paths['scenario_metadata_path']}"
    )

    if timeseries is not None:
        print(
            "Saved SD time series to: "
            f"{output_paths['timeseries_path']}"
        )
    else:
        print(
            "Long-format time-series output is "
            "disabled. Set WRITE_TIMESERIES_CSV "
            "= True in configs/common.py to "
            "create it."
        )

    print()
    print(
        f"Experiment complete: "
        f"{experiment_name}"
    )


def main():
    """
    Run the requested deterioration experiment.
    """
    arguments = parse_arguments()
    validate_run_configuration()

    selected_configurations = (
        get_selected_configurations(
            arguments.experiment
        )
    )

    print_initial_state()

    for scenario_config in (
        selected_configurations
    ):
        run_experiment(scenario_config)

    print()
    print(
        "All requested deterioration "
        "experiments are complete."
    )


if __name__ == "__main__":
    main()