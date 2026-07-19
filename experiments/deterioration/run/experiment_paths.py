"""
Define filesystem paths used by the SD deterioration exploration.

Paths are constructed relative to the location of this module so that the
experiment can be run independently of the current working directory. The
module provides paths to the experiment directory, shared source code, and 
the output directories used by the experiment.
"""

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

EXP_DIR = THIS_DIR.parent
ROOT_DIR = THIS_DIR.parents[2]
SRC_DIR = ROOT_DIR / "src"

OUTPUT_DIR = EXP_DIR / "outputs"
RAW_DIR = OUTPUT_DIR / "raw"

for directory in [
    OUTPUT_DIR,
    RAW_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)