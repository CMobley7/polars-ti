# -*- coding: utf-8 -*-
# =============================================================================
# Polars DM (Directional Movement) Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_dm(high, low, length, drift):
    """Wilder sum-smoothed +DM/-DM, matching TA-Lib PLUS_DM/MINUS_DM.

    Raw directional movement is accumulated with Wilder's running-sum recursion
    (``smoothed[i] = smoothed[i-1] - smoothed[i-1]/length + raw[i]``) seeded by
    the plain sum of the first ``length`` raw values. This is the sum-scale
    smoothing TA-Lib exposes, not the average-scale RMA of the OLD native path.
    """
    n = len(high)
    dmp = np.full(n, np.nan)
    dmn = np.full(n, np.nan)

    # Seed loop below writes at [length - 1]; guard tiny inputs.
    if n < length:
        return dmp, dmn

    pos = np.zeros(n)
    neg = np.zeros(n)
    for i in range(drift, n):
        up = high[i] - high[i - drift]
        dn = low[i - drift] - low[i]
        if up > dn and up > 0:
            pos[i] = up
        if dn > up and dn > 0:
            neg[i] = dn

    pos_sum = 0.0
    neg_sum = 0.0
    for i in range(length):
        pos_sum += pos[i]
        neg_sum += neg[i]
    dmp[length - 1] = pos_sum
    dmn[length - 1] = neg_sum

    for i in range(length, n):
        dmp[i] = dmp[i - 1] - dmp[i - 1] / length + pos[i]
        dmn[i] = dmn[i - 1] - dmn[i - 1] / length + neg[i]

    return dmp, dmn


def dm(
    high: IntoExpr,
    low: IntoExpr,
    length: int = 14,
    mamode: str = "rma",
    talib: bool = True,
    drift: int = 1,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Directional Movement (DM)

    Compares prior highs and lows to yield +DM and -DM series.
    Developed by J. Welles Wilder in 1978.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        length: Period. Default: 14
        mamode: MA type. Default: 'rma'
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        drift: The difference period for the native path. Default: 1
        offset: Shift result. Default: 0

    Returns:
        list[pl.Expr]: [DMP_14, DMN_14] expressions
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:
        _length = length

        def compute_dmp(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            high_arr = df["high"].to_numpy().astype(np.float64)
            low_arr = df["low"].to_numpy().astype(np.float64)
            from talib import PLUS_DM

            result = PLUS_DM(high_arr, low_arr, timeperiod=_length)
            return pl.Series(result)

        def compute_dmn(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            high_arr = df["high"].to_numpy().astype(np.float64)
            low_arr = df["low"].to_numpy().astype(np.float64)
            from talib import MINUS_DM

            result = MINUS_DM(high_arr, low_arr, timeperiod=_length)
            return pl.Series(result)

        struct_expr = pl.struct(high=high_expr, low=low_expr)
        dmp_expr = struct_expr.map_batches(compute_dmp, return_dtype=pl.Float64)
        dmn_expr = struct_expr.map_batches(compute_dmn, return_dtype=pl.Float64)
    else:
        # Native path: Wilder sum-smoothing to match TA-Lib PLUS_DM/MINUS_DM
        # (the OLD native path smoothed on the average scale via ma('rma'), which
        # diverged from TA-Lib by tens of points).
        _length = length
        _drift = drift

        def compute_dm_native(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            high_arr = df["high"].to_numpy().astype(np.float64)
            low_arr = df["low"].to_numpy().astype(np.float64)
            dmp_arr, dmn_arr = _nb_dm(high_arr, low_arr, _length, _drift)
            return pl.Series([{"DMP": p, "DMN": n} for p, n in zip(dmp_arr, dmn_arr)])

        struct_expr = pl.struct(high=high_expr, low=low_expr)
        fields = pl.Struct([pl.Field("DMP", pl.Float64), pl.Field("DMN", pl.Float64)])
        native = struct_expr.map_batches(compute_dm_native, return_dtype=fields)
        dmp_expr = native.struct.field("DMP")
        dmn_expr = native.struct.field("DMN")

    if offset != 0:
        dmp_expr = dmp_expr.shift(offset)
        dmn_expr = dmn_expr.shift(offset)

    return [dmp_expr.alias(f"DMP_{length}"), dmn_expr.alias(f"DMN_{length}")]
