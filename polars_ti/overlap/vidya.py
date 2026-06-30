# -*- coding: utf-8 -*-
import numpy as np
from numba import njit


@njit(cache=True)
def nb_vidya(close, abs_cmo, alpha, length):
    """Numba-optimized VIDYA calculation.

    State-dependent loop: each VIDYA value depends on previous VIDYA value.
    The recurrence is seeded at index ``length-1`` with the SMA of the first
    ``length`` closes (classic fork commit 1474768) instead of being left at 0;
    seeding from 0 produced a long, materially-wrong transient. Indices before
    ``length-1`` stay NaN.
    """
    m = close.size
    vidya = np.full(m, np.nan)
    if length - 1 < m:
        # SMA seed over the first `length` closes.
        vidya[length - 1] = close[:length].mean()

    for i in range(length, m):
        vidya[i] = alpha * abs_cmo[i] * close[i] + vidya[i - 1] * (1 - alpha * abs_cmo[i])
    return vidya


# =============================================================================
# Polars VIDYA Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def vidya(
    close: IntoExpr,
    length: int = 14,
    drift: int = 1,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Variable Index Dynamic Average (VIDYA)

    Variable Index Dynamic Average (VIDYA) was developed by Tushar Chande.
    It is similar to an EMA but it has a dynamically adjusted lookback
    period dependent on relative price volatility as measured by CMO.

    Sources:
        https://www.tradingview.com/script/hdrf0fXV-Variable-Index-Dynamic-Average-VIDYA/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: CMO period. Default: 14
        drift: Difference period for CMO. Default: 1
        talib: If True and TA-Lib available, uses TA-Lib for CMO. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: VIDYA expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _drift = drift
    _use_talib = Imports["talib"] and v_talib(talib)

    def compute_vidya(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        n = len(arr)

        # Calculate alpha
        alpha = 2.0 / (_length + 1)

        # Calculate CMO
        if _use_talib:
            from talib import CMO

            cmo_vals = CMO(arr, _length) / 100.0  # Scale to 0-1
        else:
            from polars_ti.momentum.cmo import cmo

            tmp = pl.DataFrame({"_close": arr})
            cmo_col = tmp.select(cmo("_close", length=_length, drift=_drift, talib=False)).to_series().to_numpy()
            cmo_vals = cmo_col / 100.0  # Scale to 0-1

        # Clamp |CMO| to [0, 1] and treat NaN as 0 so that ``alpha * abs_cmo``
        # stays in [0, alpha] and the recurrence is a stable convex combination
        # (a degenerate native CMO must not diverge VIDYA to +/-inf).
        abs_cmo = np.nan_to_num(np.clip(np.abs(cmo_vals), 0.0, 1.0)).astype(np.float64)

        # Use the shared Numba kernel (SMA-seeded; warmup stays NaN).
        result = nb_vidya(arr, abs_cmo, alpha, _length)

        return pl.Series(result)

    result = close_expr.map_batches(compute_vidya, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"VIDYA_{length}")
