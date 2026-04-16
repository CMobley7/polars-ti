# -*- coding: utf-8 -*-
# =============================================================================
# Polars PSAR Implementation (Numba kernel)
# =============================================================================
import numpy as np
from numba import njit
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_psar(high, low, af0, max_af):
    """Numba kernel for Parabolic SAR."""
    n = len(high)
    sar = np.zeros(n)
    long_arr = np.full(n, np.nan)
    short_arr = np.full(n, np.nan)
    af_arr = np.zeros(n)
    reversal = np.zeros(n, dtype=np.int64)

    af = af0
    af_arr[0] = af0
    af_arr[1] = af0

    # Determine initial direction
    falling = (low[0] - low[1]) > (high[1] - high[0]) and (low[0] - low[1]) > 0
    ep = low[0] if falling else high[0]
    sar[0] = high[0] if falling else low[0]

    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        if falling:
            reverse = high[i] > sar[i]
            if low[i] < ep:
                ep = low[i]
                af = min(af + af0, max_af)
            sar[i] = max(high[i - 1], sar[i])
        else:
            reverse = low[i] < sar[i]
            if high[i] > ep:
                ep = high[i]
                af = min(af + af0, max_af)
            sar[i] = min(low[i - 1], sar[i])

        if reverse:
            sar[i] = ep
            af = af0
            falling = not falling
            ep = low[i] if falling else high[i]

        if falling:
            short_arr[i] = sar[i]
        else:
            long_arr[i] = sar[i]

        af_arr[i] = af
        reversal[i] = 1 if reverse else 0

    return long_arr, short_arr, af_arr, reversal


def pl_psar(
    high: IntoExpr,
    low: IntoExpr,
    af0: float = 0.02,
    af: float = 0.02,
    max_af: float = 0.2,
    offset: int = 0,
) -> PlExpr:
    """Polars: Parabolic Stop and Reverse (PSAR)

    Determines trend direction and potential reversals using a trailing
    stop and reverse method.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        af0: Initial acceleration factor. Default: 0.02
        af: Acceleration factor step. Default: 0.02
        max_af: Maximum acceleration factor. Default: 0.2
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with PSARl, PSARs, PSARaf, PSARr columns
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    _af0 = af0 if af0 and af0 > 0 else af if af and af > 0 else 0.02

    def _compute(s: pl.Series) -> pl.Series:
        data = s.struct.unnest()
        h = data["_h"].to_numpy().astype(np.float64)
        l_ = data["_l"].to_numpy().astype(np.float64)
        long_a, short_a, af_a, rev_a = _nb_psar(h, l_, _af0, max_af)

        if offset != 0:
            for a in [long_a, short_a, af_a]:
                a[:] = np.roll(a, offset)
                if offset > 0:
                    a[:offset] = np.nan
            rev_a = np.roll(rev_a, offset)
            if offset > 0:
                rev_a[:offset] = 0

        _props = f"_{_af0}_{max_af}"
        n = len(h)
        return pl.Series(values=[
            {
                f"PSARl{_props}": long_a[i],
                f"PSARs{_props}": short_a[i],
                f"PSARaf{_props}": af_a[i],
                f"PSARr{_props}": int(rev_a[i]),
            }
            for i in range(n)
        ])

    _props = f"_{_af0}_{max_af}"
    fields = [
        pl.Field(f"PSARl{_props}", pl.Float64),
        pl.Field(f"PSARs{_props}", pl.Float64),
        pl.Field(f"PSARaf{_props}", pl.Float64),
        pl.Field(f"PSARr{_props}", pl.Int64),
    ]
    return pl.struct(
        high_expr.alias("_h"),
        low_expr.alias("_l"),
    ).map_batches(_compute, return_dtype=pl.Struct(fields)).alias(f"PSAR{_props}")

