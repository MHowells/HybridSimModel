"""
Define filesystem paths used by the gatekeeping experiment.

Paths are constructed relative to the location of this module so that the
experiment can be run independently of the current working directory. The
module provides paths to the experiment directory, shared source code, and
the output directories used for raw records, metadata, simulation arrays,
summary tables, and plots.
"""

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

EXP_DIR = THIS_DIR.parent
ROOT_DIR = THIS_DIR.parents[2]
SRC_DIR = ROOT_DIR / "src"

OUTPUT_DIR = EXP_DIR / "outputs"

RECORDS_DIR = OUTPUT_DIR / "records"
METADATA_DIR = OUTPUT_DIR / "metadata"
ARRAYS_DIR = OUTPUT_DIR / "simulation_arrays"
SUMMARY_DIR = OUTPUT_DIR / "summary_tables"
PLOTS_DIR = OUTPUT_DIR / "plots"
