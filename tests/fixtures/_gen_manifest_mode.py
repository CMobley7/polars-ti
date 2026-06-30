# -*- coding: utf-8 -*-
"""Emit the flattened AllStudy column list for ONE talib mode.

Run in a separate process per mode to avoid Numba/TA-Lib state leak:

    uv run python tests/fixtures/_gen_manifest_mode.py talib   > /tmp/cols_talib.json
    uv run python tests/fixtures/_gen_manifest_mode.py notalib > /tmp/cols_notalib.json

Prints a JSON list of sorted flattened column names to stdout.
"""

import json
import sys
import warnings

import polars as pl

sys.path.insert(0, "tests")
from _parity import flatten_structs  # noqa: E402

import polars_ti as ti  # noqa: E402,F401

SLICE_ROWS = 1500

mode = sys.argv[1]
use_talib = mode == "talib"

df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    out = df.ti.study(ti.AllStudy, cores=0, talib=use_talib, errors="ignore")

flat = flatten_structs(out)
print(json.dumps(sorted(flat.columns)))
