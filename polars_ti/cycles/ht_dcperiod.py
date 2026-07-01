# -*- coding: utf-8 -*-
# =============================================================================
# Polars HT_DCPERIOD Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def ht_dcperiod(
    close: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Hilbert Transform Dominant Cycle Period (HT_DCPERIOD)

    Uses the Hilbert Transform to identify the dominant cycle period
    in the price data, expressed in bars.

    Sources:
        John Ehlers, "Rocket Science for Traders" (2002)
        TA-Lib: HT_DCPERIOD

    Args:
        close: Column name or pl.Expr for 'close' prices.
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: HT_DCPERIOD expression (Float64).
    """
    close_expr = v_expr(close)
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    from polars_ti.cycles._ht_pipeline import nb_ht_pipeline

    _use_talib = Imports["talib"] and v_talib(talib)

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        if _use_talib:
            from talib import HT_DCPERIOD

            result = HT_DCPERIOD(arr)
        else:
            dcperiod, _, _, _, _ = nb_ht_pipeline(arr)
            result = dcperiod
            result[:32] = np.nan
        return pl.Series(values=result, name=s.name)

    result = close_expr.map_batches(_compute, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias("HT_DCPERIOD")
