# -*- coding: utf-8 -*-
# =============================================================================
# Polars Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_schaff_tc(xmacd: np.ndarray, tclength: int, factor: float):
    """Numba-accelerated Schaff Trend Cycle calculation.

    Args:
        xmacd: MACD values
        tclength: Lookback period for stochastic
        factor: Smoothing factor

    Returns:
        tuple: (pff, pf) arrays
    """
    m = len(xmacd)
    stoch1 = np.zeros(m, dtype=np.float64)
    pf = np.zeros(m, dtype=np.float64)
    stoch2 = np.zeros(m, dtype=np.float64)
    pff = np.zeros(m, dtype=np.float64)

    for i in range(1, m):
        # Calculate rolling min/max for xmacd using explicit loop
        start_idx = i - tclength + 1
        if start_idx < 0:
            start_idx = 0

        lowest_xmacd = xmacd[start_idx]
        highest_xmacd = xmacd[start_idx]
        for j in range(start_idx + 1, i + 1):
            if xmacd[j] < lowest_xmacd:
                lowest_xmacd = xmacd[j]
            if xmacd[j] > highest_xmacd:
                highest_xmacd = xmacd[j]

        xmacd_range = highest_xmacd - lowest_xmacd
        if xmacd_range == 0.0:
            xmacd_range = 1.0

        # %Fast K of MACD
        if lowest_xmacd > 0.0:
            stoch1[i] = 100.0 * (xmacd[i] - lowest_xmacd) / xmacd_range
        else:
            stoch1[i] = stoch1[i - 1]

        # Smoothed % Fast D of MACD
        pf[i] = pf[i - 1] + factor * (stoch1[i] - pf[i - 1])

        # Find min and max of pf so far
        pf_start = i - tclength + 1
        if pf_start < 0:
            pf_start = 0

        lowest_pf = pf[pf_start]
        highest_pf = pf[pf_start]
        for j in range(pf_start + 1, i + 1):
            if pf[j] < lowest_pf:
                lowest_pf = pf[j]
            if pf[j] > highest_pf:
                highest_pf = pf[j]

        pf_range = highest_pf - lowest_pf
        if pf_range == 0.0:
            pf_range = 1.0

        # % of Fast K of PF
        if pf_range > 0.0:
            stoch2[i] = 100.0 * (pf[i] - lowest_pf) / pf_range
        else:
            stoch2[i] = stoch2[i - 1]

        # Final smoothed value
        pff[i] = pff[i - 1] + factor * (stoch2[i] - pff[i - 1])

    return pff, pf


def stc(
    close: IntoExpr = "close",
    tclength: int = 10,
    fast: int = 12,
    slow: int = 26,
    factor: float = 0.5,
    offset: int = 0,
) -> PlExpr:
    """Polars: Schaff Trend Cycle (STC)

    The Schaff Trend Cycle is an evolution of MACD incorporating
    two cascaded stochastic calculations with additional smoothing.

    Sources:
        https://www.prorealcode.com/prorealtime-indicators/schaff-trend-cycle2/

    Args:
        close (IntoExpr): Column name or expression for 'close'. Default: "close"
        tclength (int): SchaffTC Signal-Line length. Default: 10
        fast (int): The short EMA period. Default: 12
        slow (int): The long EMA period. Default: 26
        factor (float): Smoothing factor. Default: 0.5
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pl.Expr: Struct expression with STC, STCmacd, STCstoch columns
    """
    from polars_ti.overlap.ema import ema

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    if slow < fast:
        fast, slow = slow, fast

    _length = max(tclength, fast, slow)
    _props = f"_{tclength}_{fast}_{slow}_{factor}"

    # Calculate MACD (fast EMA - slow EMA)
    fast_ema = ema(close_expr, length=fast)
    slow_ema = ema(close_expr, length=slow)
    xmacd = fast_ema - slow_ema

    def compute_stc(s: pl.Series) -> pl.Series:
        """Compute STC using Numba kernel."""
        macd_arr = s.to_numpy().astype(np.float64)
        pff, pf = nb_schaff_tc(macd_arr, tclength, factor)

        # Set warmup period to NaN
        pff[: _length - 1] = np.nan
        pf[: _length - 1] = np.nan

        if offset != 0:
            pff = np.roll(pff, offset)
            pf = np.roll(pf, offset)
            macd_arr = np.roll(macd_arr, offset)
            if offset > 0:
                pff[:offset] = np.nan
                pf[:offset] = np.nan
                macd_arr[:offset] = np.nan
            else:
                pff[offset:] = np.nan
                pf[offset:] = np.nan
                macd_arr[offset:] = np.nan

        return pl.DataFrame(
            {
                f"STC{_props}": pff,
                f"STCmacd{_props}": macd_arr,
                f"STCstoch{_props}": pf,
            }
        ).to_struct(f"STC{_props}")

    return_dtype = pl.Struct(
        [
            pl.Field(f"STC{_props}", pl.Float64),
            pl.Field(f"STCmacd{_props}", pl.Float64),
            pl.Field(f"STCstoch{_props}", pl.Float64),
        ]
    )

    return xmacd.map_batches(compute_stc, return_dtype=return_dtype).alias(f"STC{_props}")
