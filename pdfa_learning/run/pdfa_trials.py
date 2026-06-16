"""Run repeated ALERGIA trials for subspecialty pathway subsets.

This module loads pathway sequences for one or more orthopaedic
subspecialty-severity groups, learns a PDFA for each alpha and random seed,
evaluates each learned model, and saves the results using the same filenames,
column names, and column order as the original notebook.
"""

from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import pdfa_learning as pl


MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIRECTORY.parent

DEFAULT_DATA_ROOT = PROJECT_ROOT / "csv"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "csv"
DEFAULT_ALPHABET_ROOT = PROJECT_ROOT / "pkl"

PATHWAY_COLUMN = "PDFA Pathway ExPO"

EVALUATION_COLUMN_MAP = {
    "Mean Absolute Distance": "mean_absolute_distance",
    "Jensen-Shannon Divergence": "jensen_shannon_divergence",
    "Coverage Prop.": "sequence_coverage",
    "Coverage Mass.": "probability_mass_coverage",
    "Train Set PDFA Mass": "training_sequence_pdfa_mass",
    "Completely New Mass": "outside_training_sequence_pdfa_mass",
}

BASE_COLUMNS = [
    "Subspec",
    "Alpha",
    "Seed",
    "Mean Absolute Distance",
    "Jensen-Shannon Divergence",
    "Coverage Prop.",
    "Coverage Mass.",
    "Train Set PDFA Mass",
    "Completely New Mass",
    "ALERGIA Runtime Seconds",
    "ALERGIA Runtime Minutes",
]

TRACKING_COLUMNS = [
    "Initial States",
    "Final States",
    "Attempted Merges",
    "Successful Merges",
    "Recursive Merge Attempts",
    "Recursive Merge Failures",
]

RESULT_COLUMNS = BASE_COLUMNS + TRACKING_COLUMNS


def load_pathway_sequences(
    csv_path,
    *,
    pathway_column=PATHWAY_COLUMN,
):
    """Load and clean pathway sequences from one input CSV file."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Could not find the pathway data file: {csv_path}"
        )

    data = pd.read_csv(csv_path)

    if pathway_column not in data.columns:
        raise ValueError(
            f"{csv_path} does not contain the required "
            f"column {pathway_column!r}."
        )

    pathways = data[pathway_column].dropna().copy()

    pathways = pathways[
        ~pathways.str.contains("Y", na=False)
    ]
    pathways = pathways.str.replace("X", "", regex=False)
    pathways = pathways.str.replace("Z", "", regex=False)
    pathways = pathways[pathways != ""]

    sequences = pathways.tolist()

    if not sequences:
        raise ValueError(
            f"No valid pathway sequences remained after cleaning "
            f"{csv_path}."
        )

    return sequences


def load_existing_results(
    save_path,
):
    """Load an existing results file or create an empty result table."""
    save_path = Path(save_path)

    if save_path.is_file():
        results = pd.read_csv(save_path)

        for column in RESULT_COLUMNS:
            if column not in results.columns:
                results[column] = np.nan

        return sort_results(results)

    return pd.DataFrame(columns=RESULT_COLUMNS)


def sort_results(results):
    """Apply the established output column order and row sorting."""
    return (
        results.reindex(columns=RESULT_COLUMNS)
        .sort_values(by=["Seed", "Alpha"])
        .reset_index(drop=True)
    )


def trial_already_exists(
    results,
    *,
    subspec,
    alpha,
    seed,
):
    """Return whether a subspecialty-alpha-seed result already exists."""
    return bool(
        (
            (results["Subspec"] == subspec)
            & (results["Alpha"] == alpha)
            & (results["Seed"] == seed)
        ).any()
    )


def write_csv_safely(
    dataframe,
    path,
    *,
    index=False,
):
    """Write a CSV through a temporary file before replacing the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")
    dataframe.to_csv(temporary_path, index=index)
    temporary_path.replace(path)


def write_pickle_safely(
    value,
    path,
):
    """Write a pickle through a temporary file before replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(value, file)

    temporary_path.replace(path)


def prepare_and_save_alphabet(
    input_path,
    alphabet_path,
    *,
    pathway_column=PATHWAY_COLUMN,
):
    """Create and save the alphabet for one subspecialty dataset."""
    sequences = load_pathway_sequences(
        input_path,
        pathway_column=pathway_column,
    )
    alphabet = pl.get_alphabet(sequences)

    write_pickle_safely(
        alphabet,
        alphabet_path,
    )

    return alphabet


def create_result_row(
    *,
    subspec,
    alpha,
    seed,
    evaluation,
    tracking,
    runtime_seconds,
):
    """Construct one result row using the established CSV schema."""
    return {
        "Subspec": subspec,
        "Alpha": alpha,
        "Seed": seed,
        "Mean Absolute Distance": evaluation[
            EVALUATION_COLUMN_MAP[
                "Mean Absolute Distance"
            ]
        ],
        "Jensen-Shannon Divergence": evaluation[
            EVALUATION_COLUMN_MAP[
                "Jensen-Shannon Divergence"
            ]
        ],
        "Coverage Prop.": evaluation[
            EVALUATION_COLUMN_MAP["Coverage Prop."]
        ],
        "Coverage Mass.": evaluation[
            EVALUATION_COLUMN_MAP["Coverage Mass."]
        ],
        "Train Set PDFA Mass": evaluation[
            EVALUATION_COLUMN_MAP[
                "Train Set PDFA Mass"
            ]
        ],
        "Completely New Mass": evaluation[
            EVALUATION_COLUMN_MAP[
                "Completely New Mass"
            ]
        ],
        "ALERGIA Runtime Seconds": runtime_seconds,
        "ALERGIA Runtime Minutes": runtime_seconds / 60,
        "Initial States": tracking["initial_states"],
        "Final States": tracking["final_states"],
        "Attempted Merges": tracking["attempted_merges"],
        "Successful Merges": tracking["successful_merges"],
        "Recursive Merge Attempts": tracking[
            "recursive_merge_attempts"
        ],
        "Recursive Merge Failures": tracking[
            "recursive_merge_failures"
        ],
    }


def run_one_pdfa_seed_from_config(
    seed_config,
):
    """Run all outstanding alpha values for one subspecialty and seed.

    This function is designed to be called by ``multiprocessing.Pool``. 
    It reads the pathway data, creates one train-test split, constructs 
    the initial PPTA once, and then runs ALERGIA for each alpha supplied 
    in ``seed_config``.

    Parameters
    ----------
    seed_config : mapping
        Serialisable configuration containing ``subspec``, ``seed``,
        ``alpha_values``, ``input_path``, ``alphabet``, ``pathway_column``,
        ``test_size``, and ``method``.

    Returns
    -------
    dict
        Completed result rows and identifying metadata.
    """
    subspec = str(seed_config["subspec"])
    seed = int(seed_config["seed"])
    alpha_values = [
        float(alpha)
        for alpha in seed_config["alpha_values"]
    ]
    input_path = Path(str(seed_config["input_path"]))
    alphabet = list(seed_config["alphabet"])
    pathway_column = str(seed_config["pathway_column"])
    test_size = float(seed_config["test_size"])
    method = str(seed_config["method"])

    sequences = load_pathway_sequences(
        input_path,
        pathway_column=pathway_column,
    )

    train_sequences, test_sequences = train_test_split(
        sequences,
        test_size=test_size,
        random_state=seed,
    )

    initial_matrix = pl.get_transition_matrix(
        train_sequences,
        alphabet,
    )
    initial_states = pl.get_initial_states(
        train_sequences,
    )

    result_rows = []

    for alpha in alpha_values:
        start_time = time.perf_counter()

        (
            final_matrix,
            final_states,
            tracking,
        ) = pl.alergia(
            initial_matrix,
            initial_states,
            alphabet,
            alpha,
            method=method,
        )

        runtime_seconds = time.perf_counter() - start_time

        probability_matrix = pl.probability_transition_matrix(
            final_matrix,
            final_states,
            alphabet,
        )

        evaluation = pl.evaluate_pdfa(
            test_sequences=test_sequences,
            train_sequences=train_sequences,
            probability_matrix=probability_matrix,
            alphabet=alphabet,
        )

        result_rows.append(
            create_result_row(
                subspec=subspec,
                alpha=alpha,
                seed=seed,
                evaluation=evaluation,
                tracking=tracking,
                runtime_seconds=runtime_seconds,
            )
        )

    return {
        "subspec": subspec,
        "seed": seed,
        "alpha_values": alpha_values,
        "rows": result_rows,
    }


def upsert_pdfa_result_rows(
    rows,
    save_path,
):
    """Add or replace completed rows in one subspecialty results CSV.

    Existing rows with the same ``Subspec``, ``Alpha``, and ``Seed`` are 
    replaced before the file is sorted and safely rewritten.

    Parameters
    ----------
    rows : sequence of mappings
        Completed result rows.
    save_path : str or pathlib.Path
        Per-subspecialty result CSV path.

    Returns
    -------
    pandas.DataFrame
        Updated, ordered results table.
    """
    save_path = Path(save_path)
    results = load_existing_results(save_path)

    for row in rows:
        same_trial = (
            (results["Subspec"] == row["Subspec"])
            & (results["Alpha"] == row["Alpha"])
            & (results["Seed"] == row["Seed"])
        )

        results = results.loc[~same_trial]

    if rows:
        new_rows = (
            pd.DataFrame(rows)
            .reindex(columns=RESULT_COLUMNS)
        )

        if results.empty:
            results = new_rows.copy()
        else:
            results = pd.concat(
                [results, new_rows],
                ignore_index=True,
            )

    results = sort_results(results)
    write_csv_safely(
        results,
        save_path,
        index=False,
    )

    return results
