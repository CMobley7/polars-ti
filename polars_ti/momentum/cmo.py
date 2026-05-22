# -*- coding: utf-8 -*-
# =============================================================================
# Polars CMO Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_cmo(
    close: IntoExpr,
    length: int = 14,
    scalar: float = 100.0,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Chande Momentum Oscillator (CMO)

    Measures momentum with overbought at 50 and oversold at -50.
    CMO = scalar * (sum_gains - sum_losses) / (sum_gains + sum_losses)

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/chande-momentum-oscillator-cmo/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling period. Default: 14
        scalar: Multiplication factor. Default: 100
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: CMO expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:
        _length = length

        def compute_cmo_talib(s: pl.Series) -> pl.Series:
            from talib import CMO as TALIB_CMO

            arr = s.to_numpy().astype(np.float64)
            result = TALIB_CMO(arr, timeperiod=_length)
            return pl.Series(result)

        cmo_expr = close_expr.map_batches(compute_cmo_talib, return_dtype=pl.Float64)
    else:
        # Calculate momentum (diff)
        mom = close_expr.diff(1)

        # Positive gains (clipped lower=0)
        pos = mom.clip(lower_bound=0)

        # Negative losses (clipped upper=0, then abs)
        neg = mom.clip(upper_bound=0).abs()

        # Rolling sums
        pos_sum = pos.rolling_sum(window_size=length)
        neg_sum = neg.rolling_sum(window_size=length)

        # CMO = scalar * (pos_sum - neg_sum) / (pos_sum + neg_sum)
        total = pos_sum + neg_sum
        cmo_expr = pl.when(total != 0).then(scalar * (pos_sum - neg_sum) / total).otherwise(pl.lit(None))

    if offset != 0:
        cmo_expr = cmo_expr.shift(offset)

    return cmo_expr.alias(f"CMO_{length}")
