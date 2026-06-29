# -*- coding: utf-8 -*-
"""Generate golden OLD-library reference fixtures for the parity oracle.

The OLD (pandas) Polars-TI library is the reference implementation. This script
runs the OLD ``AllStudy`` for both ``talib=True`` and ``talib=False`` over a
deterministic slice of SPY_D and writes the result to Parquet.

These parquets are the committed oracle that the new (Polars) library is graded
against. Re-running this script reproduces the parquets byte-for-byte (the input
slice is deterministic; no timestamps or randomness are introduced).

WHY A SUBPROCESS / SEPARATE VENV
--------------------------------
Both the old and the new libraries import as ``polars_ti``. The new library is
editable-installed at the repo root, so importing from the repo root resolves to
the NEW package. The old library lives under ``tmp/polars-ti`` and MUST be run
from its own venv AND from a neutral cwd so ``import polars_ti`` resolves to the
old tree. This script asserts ``polars_ti.__file__`` points into ``tmp/polars-ti``
before trusting its output.

EXACT COMMAND TO REGENERATE (run from a neutral cwd, e.g. /tmp):

    cd /tmp && \
    /home/cmobley/Documents/Projects/polars-ti/tmp/polars-ti/.venv/bin/python \
        /home/cmobley/Documents/Projects/polars-ti/tests/fixtures/_generate_old.py

Outputs:
    tests/fixtures/old_talib.parquet     (OLD AllStudy, talib=True)
    tests/fixtures/old_notalib.parquet   (OLD AllStudy, talib=False)
"""

import warnings

import pandas as pd

import polars_ti as ti

warnings.simplefilter("ignore")

ROOT = "/home/cmobley/Documents/Projects/polars-ti"
DATA = f"{ROOT}/data/SPY_D.csv"
OUT = f"{ROOT}/tests/fixtures"

# Deterministic fixture slice: the first 1500 rows of SPY_D. This comfortably
# covers the warmup of the longest default lookbacks (~610 bars) while keeping
# the parquet small (a few MB, not ~18MB for the full 7192-bar set).
SLICE_ROWS = 1500

# Guard: make sure we are running the OLD library, not the NEW one.
assert "tmp/polars-ti" in ti.__file__, (
    f"Expected the OLD polars_ti under tmp/polars-ti, got: {ti.__file__}. "
    "Run this from a neutral cwd (e.g. /tmp) with the OLD venv python."
)
print("OLD polars_ti:", ti.__file__)


def load():
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()
    return df.head(SLICE_ROWS)


for talib in (True, False):
    tag = "talib" if talib else "notalib"
    df = load()
    n0 = df.shape[1]
    df.ti.study(ti.AllStudy, cores=0, talib=talib)
    added = df.shape[1] - n0
    # Reset index so the date becomes a regular column.
    out = df.reset_index()
    out.to_parquet(f"{OUT}/old_{tag}.parquet")
    print(f"OLD {tag}: rows={df.shape[0]} cols={df.shape[1]} added={added}")

print("done old")
