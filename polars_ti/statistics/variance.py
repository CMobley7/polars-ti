import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars VARIANCE Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.utils._rolling import rolling_variance
from polars_ti.utils._validate import v_expr


def variance(
    close: IntoExpr,
    length: int = 30,
    ddof: int = 0,
    talib: bool = True,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Variance

    Calculates rolling variance using compensated moments or explicit TA-Lib dispatch.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        ddof: Delta Degrees of Freedom (population std, TA-Lib/TradingView convention). Default: 0
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Variance expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    if Imports["talib"] and v_talib(talib) and ddof == 0:

        def compute_var(s: pl.Series) -> pl.Series:
            from talib import VAR

            arr = s.to_numpy().astype(np.float64)
            return pl.Series(VAR(arr, timeperiod=length))

        result = close_expr.map_batches(compute_var, return_dtype=pl.Float64)
    else:
        if length < 1 or not 0 <= ddof <= 255:
            raise ValueError("positive length and ddof in [0, 255] required")

        def compute_native(s: pl.Series) -> pl.Series:
            """Evaluate stable variance using centered compensated moments."""
            values = s.to_numpy().astype(np.float64)
            result = rolling_variance(values, length, ddof)
            output = pl.Series(result)
            if length <= ddof:
                return output.fill_nan(None)
            missing = s.is_null().cast(pl.Int64).rolling_sum(length, min_samples=length)
            return output.set(missing.is_null() | (missing > 0), None)

        result = close_expr.map_batches(compute_native, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"VAR_{length}")
