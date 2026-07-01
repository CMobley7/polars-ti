# -*- coding: utf-8 -*-
# =============================================================================
# Polars HT_DCPHASE Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def ht_dcphase(
    close: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Hilbert Transform Dominant Cycle Phase (HT_DCPHASE)

    Uses the Hilbert Transform to measure the current phase (in degrees)
    of the dominant cycle in the price data.

    Sources:
        John Ehlers, "Rocket Science for Traders" (2002)
        TA-Lib: HT_DCPHASE

    Args:
        close: Column name or pl.Expr for 'close' prices.
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: HT_DCPHASE expression (Float64, degrees).
    """
    close_expr = v_expr(close)
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    from polars_ti.cycles._ht_pipeline import nb_ht_pipeline

    _use_talib = Imports["talib"] and v_talib(talib)

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        if _use_talib:
            from talib import HT_DCPHASE

            result = HT_DCPHASE(arr)
        else:
            _, dcphase, _, _, _ = nb_ht_pipeline(arr)
            result = dcphase
            result[:63] = np.nan
        return pl.Series(values=result, name=s.name)

    result = close_expr.map_batches(_compute, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias("HT_DCPHASE")
