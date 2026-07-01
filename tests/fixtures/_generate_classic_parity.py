# -*- coding: utf-8 -*-
"""Generate committed golden fixtures for classic-fork parity tests.

The Tulip-parity indicators (``tests/test_tulip_parity.py``) and ``smc_sweep``
(``tests/momentum/test_smc_sweep_polars.py``) validate polars_ti native output
against the ``pandas_ta_classic`` reference fork cloned at
``tmp/pandas-ta-classic``.  That clone is gitignored and NOT present in CI, so
those cross-validations previously just SKIPPED there — the indicators went
unvalidated in CI.

This script runs the classic fork ONCE over a deterministic SPY_D slice and
writes the expected reference outputs to compact committed parquet fixtures:

    tests/fixtures/classic_tulip_parity.parquet   (tulip-parity indicators)
    tests/fixtures/classic_smc_sweep.parquet      (smc_sweep parameter matrix)

The parity tests then assert against these committed fixtures, so full
validation runs in CI without the clone.  (TA-Lib assertions are kept
separately in the tests where TA-Lib has the function.)

EXACT COMMAND TO REGENERATE (from the repo root, with the clone present):

    uv run python tests/fixtures/_generate_classic_parity.py

Re-running reproduces the parquets deterministically (fixed SPY_D slice, no
randomness or timestamps introduced).
"""

import os
import sys

import numpy as np
import polars as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIC = os.path.join(REPO_ROOT, "tmp", "pandas-ta-classic")
FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures")
DATA_CSV = os.path.join(REPO_ROOT, "data", "SPY_D.csv")

SLICE_ROWS = 500
SMC_ROWS = 1500
SMC_PARAMS = [(15, 1.5), (10, 2.0), (20, 1.0), (20, 2.5)]


def main() -> None:
    if not os.path.isdir(CLASSIC):
        raise SystemExit(f"classic fork not found at {CLASSIC}")
    sys.path.insert(0, CLASSIC)
    import pandas as pd
    import pandas_ta_classic as pta

    # ---- Tulip-parity fixtures (SLICE_ROWS rows) -------------------------
    df = pl.read_csv(DATA_CSV, try_parse_dates=True).head(SLICE_ROWS)
    a = {c: df[c].to_numpy().astype(float) for c in ["open", "high", "low", "close", "volume"]}
    o, h, lo, c, v = (pd.Series(a[k]) for k in ["open", "high", "low", "close", "volume"])

    cols: dict[str, np.ndarray] = {}
    cols["avgprice"] = pta.avgprice(o, h, lo, c, talib=False).values
    cols["medprice"] = pta.medprice(h, lo, talib=False).values
    cols["typprice"] = pta.typprice(h, lo, c, talib=False).values
    cols["cvi"] = pta.cvi(h, lo).values
    cols["hvol"] = pta.hvol(c).values
    cols["avolume"] = pta.avolume(c).values
    cols["marketfi"] = pta.marketfi(h, lo, v).values
    cols["vosc"] = pta.vosc(v).values
    cols["wad"] = pta.wad(h, lo, c).values
    cols["emv"] = pta.emv(h, lo, v).values
    cols["fosc"] = pta.fosc(c).values
    cols["stderr"] = pta.stderr(c).values
    cols["md"] = pta.md(c).values

    msw = pta.msw(c)
    cols["msw_sine"] = msw["MSW_SINE_5"].values
    cols["msw_lead"] = msw["MSW_LEAD_5"].values

    out = pl.DataFrame({k: np.asarray(vv, dtype=float) for k, vv in cols.items()})
    out_path = os.path.join(FIXTURES, "classic_tulip_parity.parquet")
    out.write_parquet(out_path)
    print(f"wrote {out_path}  shape={out.shape}")

    # ---- smc_sweep fixtures (SMC_ROWS rows, parameter matrix) ------------
    dfs = pl.read_csv(DATA_CSV, try_parse_dates=True).head(SMC_ROWS)
    so = pd.Series(dfs["open"].to_numpy())
    sh = pd.Series(dfs["high"].to_numpy())
    sl = pd.Series(dfs["low"].to_numpy())
    sc = pd.Series(dfs["close"].to_numpy())
    smc_cols: dict[str, np.ndarray] = {}
    for length, wick_mult in SMC_PARAMS:
        ref = pta.smc_sweep(so, sh, sl, sc, length=length, wick_mult=wick_mult).to_numpy()
        smc_cols[f"smc_{length}_{wick_mult}"] = np.asarray(ref, dtype=float)
    smc_out = pl.DataFrame(smc_cols)
    smc_path = os.path.join(FIXTURES, "classic_smc_sweep.parquet")
    smc_out.write_parquet(smc_path)
    print(f"wrote {smc_path}  shape={smc_out.shape}")


if __name__ == "__main__":
    main()
