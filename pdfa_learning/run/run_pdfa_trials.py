"""Run the complete subspecialty PDFA experiment."""

import multiprocessing
from pathlib import Path

from pdfa_trials import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_ALPHABET_ROOT,
    PATHWAY_COLUMN,
    load_existing_results,
    prepare_and_save_alphabet,
    run_one_pdfa_seed_from_config,
    trial_already_exists,
    upsert_pdfa_result_rows,
)


SUBSPECS = [
    "foot_ankle_low",
    "foot_ankle_medium",
    "foot_ankle_high",
    "hand_low",
    "hand_medium",
    "hand_high",
    "hip_low",
    "hip_medium",
    "hip_high",
    "knee_low",
    "knee_medium",
    "knee_high",
    "paeds_low",
    "paeds_medium",
    "paeds_high",
    "shoulder_elbow_low",
    "shoulder_elbow_medium",
    "shoulder_elbow_high",
    "spine_low",
    "spine_medium",
    "spine_high",
]

ALPHA_VALUES = [
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

SEED_VALUES = list(range(30))

DATA_FOLDER = "pathways"
SAVE_FOLDER = "pdfa_trial_results"
ALPHABET_FOLDER = "alphabets"

DATA_ROOT = DEFAULT_DATA_ROOT
OUTPUT_ROOT = DEFAULT_RESULTS_ROOT
ALPHABET_ROOT = DEFAULT_ALPHABET_ROOT

PATHWAY_COLUMN_NAME = PATHWAY_COLUMN
TEST_SIZE = 0.2
ALERGIA_METHOD = "de_la_higuera"

N_WORKERS = 4


def choose_n_workers(n_jobs):
    """Choose a worker count for the requested jobs."""
    cpu_count = multiprocessing.cpu_count()
    return min(n_jobs, max(1, cpu_count - 2))


def validate_run_configuration():
    """Validate the multiprocessing experiment configuration."""
    if not SUBSPECS:
        raise ValueError(
            "SUBSPECS must contain at least one subspecialty."
        )

    if len(set(SUBSPECS)) != len(SUBSPECS):
        raise ValueError(
            "SUBSPECS must not contain duplicate names."
        )

    if not ALPHA_VALUES:
        raise ValueError(
            "ALPHA_VALUES must contain at least one value."
        )

    if len(set(ALPHA_VALUES)) != len(ALPHA_VALUES):
        raise ValueError(
            "ALPHA_VALUES must not contain duplicates."
        )
    
    if any(
        not 0 < float(alpha) <= 2
        for alpha in ALPHA_VALUES
    ):
        raise ValueError(
            "Every alpha value must lie in the interval (0, 2]."
        )

    if not SEED_VALUES:
        raise ValueError(
            "SEED_VALUES must contain at least one seed."
        )

    if len(set(SEED_VALUES)) != len(SEED_VALUES):
        raise ValueError(
            "SEED_VALUES must not contain duplicates."
        )

    if not 0 < TEST_SIZE < 1:
        raise ValueError(
            "TEST_SIZE must be strictly between zero and one."
        )

    if N_WORKERS is not None and int(N_WORKERS) < 1:
        raise ValueError(
            "N_WORKERS must be at least 1 or None."
        )


def prepare_pdfa_seed_configs(
    *,
    data_directory,
    output_directory,
    alphabet_directory,
):
    """Prepare all outstanding subspecialty-seed jobs.

    Existing subspecialty-alpha-seed combinations are omitted. The alphabet
    for each subspecialty is created and saved once by the main process.
    """
    seed_configs = []

    for subspec in SUBSPECS:
        input_path = data_directory / f"{subspec}.csv"
        save_path = (
            output_directory
            / f"{subspec}_analysis_results.csv"
        )
        alphabet_path = (
            alphabet_directory
            / f"{subspec}_alphabet.pkl"
        )

        alphabet = prepare_and_save_alphabet(
            input_path,
            alphabet_path,
            pathway_column=PATHWAY_COLUMN_NAME,
        )

        existing_results = load_existing_results(save_path)

        for seed in SEED_VALUES:
            outstanding_alphas = [
                alpha
                for alpha in ALPHA_VALUES
                if not trial_already_exists(
                    existing_results,
                    subspec=subspec,
                    alpha=alpha,
                    seed=seed,
                )
            ]

            if not outstanding_alphas:
                print(
                    f"Skipping {subspec}, seed={seed}; "
                    "all alpha values already exist."
                )
                continue

            seed_configs.append(
                {
                    "subspec": subspec,
                    "seed": seed,
                    "alpha_values": outstanding_alphas,
                    "input_path": str(input_path),
                    "alphabet": alphabet,
                    "pathway_column": PATHWAY_COLUMN_NAME,
                    "test_size": TEST_SIZE,
                    "method": ALERGIA_METHOD,
                }
            )

    return seed_configs


def main():
    """Run all outstanding PDFA seed jobs in one shared process pool."""
    validate_run_configuration()

    data_directory = Path(DATA_ROOT) / DATA_FOLDER
    output_directory = Path(OUTPUT_ROOT) / SAVE_FOLDER
    alphabet_directory = Path(ALPHABET_ROOT) / ALPHABET_FOLDER

    if not data_directory.is_dir():
        raise FileNotFoundError(
            f"Input data directory does not exist: {data_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Reading pathway data from: {data_directory}")
    print(f"Saving PDFA results to: {output_directory}")
    print(f"Saving alphabets to: {alphabet_directory}")
    seed_configs = prepare_pdfa_seed_configs(
        data_directory=data_directory,
        output_directory=output_directory,
        alphabet_directory=alphabet_directory,
    )
    n_jobs = len(seed_configs)

    if n_jobs == 0:
        print()
        print("All requested PDFA trials were already complete.")
        return

    if N_WORKERS is None:
        n_workers = choose_n_workers(n_jobs)
    else:
        n_workers = min(int(N_WORKERS), n_jobs)

    print()
    print(
        "CPU count visible to Python: "
        f"{multiprocessing.cpu_count()}"
    )
    print(
        f"Running {n_jobs} subspecialty-seed job(s) "
        f"using {n_workers} worker process(es)."
    )

    with multiprocessing.Pool(
        processes=n_workers,
    ) as pool:
        completed_jobs = pool.imap_unordered(
            run_one_pdfa_seed_from_config,
            seed_configs,
        )

        for completed_count, completed_job in enumerate(
            completed_jobs,
            start=1,
        ):
            subspec = completed_job["subspec"]
            seed = completed_job["seed"]
            rows = completed_job["rows"]

            save_path = (
                output_directory
                / f"{subspec}_analysis_results.csv"
            )

            upsert_pdfa_result_rows(
                rows,
                save_path,
            )

            completed_alphas = ", ".join(
                str(alpha)
                for alpha in completed_job["alpha_values"]
            )

            alergia_runtime_seconds = sum(
                float(row["ALERGIA Runtime Seconds"])
                for row in rows
            )

            print(
                f"Recorded {subspec}, seed={seed}, "
                f"alpha(s)={completed_alphas}, "
                f"ALERGIA runtime="
                f"{alergia_runtime_seconds:.2f} seconds "
                f"({completed_count}/{n_jobs})."
            )

    print()
    print("All requested PDFA trials are complete.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
