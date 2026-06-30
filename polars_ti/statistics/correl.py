# -*- coding: utf-8 -*-
# =============================================================================
# Polars CORREL (Pearson Correlation Coefficient) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def correl(
    close: IntoExpr,
    benchmark: IntoExpr,
    length: int = 30,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Pearson Correlation Coefficient (CORREL)

    Measures the linear relationship between two series over a rolling window.
    Values range from -1 (perfect negative) to +1 (perfect positive).

    Matches TA-Lib ``CORREL`` (rolling Pearson correlation of the raw price
    series, not returns).

    CORREL = rolling Pearson correlation of close and benchmark over `length`.

    Args:
        close: Column name or pl.Expr for the security 'close' prices
        benchmark: Column name or pl.Expr for the benchmark prices
        length: Rolling window period. Default: 30
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: CORREL expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    benchmark_expr = v_expr(benchmark)
    if close_expr is None or benchmark_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length

    if _use_talib:

        def compute_correl(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            c = df["_c"].to_numpy().astype(np.float64)
            b = df["_b"].to_numpy().astype(np.float64)
            from talib import CORREL as TALIB_CORREL

            return pl.Series(TALIB_CORREL(c, b, timeperiod=_length))

        struct_expr = pl.struct(close_expr.alias("_c"), benchmark_expr.alias("_b"))
        correl_expr = struct_expr.map_batches(compute_correl, return_dtype=pl.Float64)
    else:
        correl_expr = pl.rolling_corr(close_expr, benchmark_expr, window_size=length, min_samples=length)

    if offset != 0:
        correl_expr = correl_expr.shift(offset)

    return correl_expr.alias(f"CORREL_{length}")
