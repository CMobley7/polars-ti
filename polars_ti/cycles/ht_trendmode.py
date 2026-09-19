# -*- coding: utf-8 -*-
# =============================================================================
# Polars HT_TRENDMODE Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.cycles._ht_utils import run_hilbert
from polars_ti.utils._validate import v_expr


def ht_trendmode(
    close: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Hilbert Transform Trend vs Cycle Mode (HT_TRENDMODE)

    Uses the Hilbert Transform to classify each bar as either trending (1)
    or cycling (0). A value of 1 indicates the market is in a trend phase;
    0 indicates a cycle (oscillating) phase.

    Sources:
        John Ehlers, "Rocket Science for Traders" (2002)
        TA-Lib: HT_TRENDMODE

    Args:
        close: Column name or pl.Expr for 'close' prices.
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: HT_TRENDMODE expression (nullable Int32, values 0 or 1).
    """
    close_expr = v_expr(close)
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    from polars_ti.cycles._ht_pipeline import nb_ht_pipeline

    _use_talib = Imports["talib"] and v_talib(talib)

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        if _use_talib:
            from talib import HT_TRENDMODE

            result = run_hilbert(arr, lambda inputs: (HT_TRENDMODE(inputs[0]),), 0)[0]
        else:
            result = run_hilbert(arr, lambda inputs: (nb_ht_pipeline(inputs[0])[4],), 0)[0]

        return pl.Series(values=result, name=s.name, nan_to_null=True).cast(pl.Int32)

    result = close_expr.map_batches(_compute, return_dtype=pl.Int32)

    if offset != 0:
        result = result.shift(offset)

    return result.alias("HT_TRENDMODE")
