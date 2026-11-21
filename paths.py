"""Where the data, the processed inputs and the Kaggle run outputs live.

Every script in this repository imports its folders from here. Point the environment variables at your own copies, or keep
the defaults (folders inside workspace/ next to this file). data/README.md lists what each folder holds.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.environ.get("BSISO_WORKSPACE", os.path.join(ROOT, "workspace"))
DATA = os.environ.get("BSISO_DATA", os.path.join(WORKSPACE, "data"))                        # raw downloads (IMD, ERA5, NCEP, OLR, ECMWF S2S, indices)
WORK = os.environ.get("BSISO_WORK", os.path.join(WORKSPACE, "intermediate"))                # processed model inputs and result tables
RUNS = os.environ.get("BSISO_RUNS", os.path.join(WORKSPACE, "runs"))                        # Kaggle outputs: runs/<ROUND>/out/out/*.pkl
KDATA = os.environ.get("BSISO_KAGGLE_DATASET", os.path.join(WORKSPACE, "kaggle_dataset"))   # the files of the Kaggle dataset profa-inputs-ext
