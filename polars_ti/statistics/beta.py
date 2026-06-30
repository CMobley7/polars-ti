# -*- coding: utf-8 -*-
# =============================================================================
# Polars BETA Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def beta(
    close: IntoExpr,
    benchmark: IntoExpr,
    length: int = 30,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Beta (BETA)

    Beta measures the sensitivity of a security's returns to the returns of a
    benchmark. A beta of 1 means the security moves with the benchmark; above 1
    means more volatile, below 1 means less volatile.

    Matches TA-Lib ``BETA(real0=close, real1=benchmark)``: the rolling slope of
    the *benchmark* returns regressed on the *close* returns over ``length``
    return observations (so the variance denominator uses ``close`` returns).

    BETA = Cov(close_ret, bench_ret) / Var(close_ret)

    Args:
        close: Column name or pl.Expr for the security 'close' prices
        benchmark: Column name or pl.Expr for the benchmark prices
        length: Rolling window of returns. Default: 30
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: BETA expression
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

        def compute_beta(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            c = df["_c"].to_numpy().astype(np.float64)
            b = df["_b"].to_numpy().astype(np.float64)
            from talib import BETA as TALIB_BETA

            return pl.Series(TALIB_BETA(c, b, timeperiod=_length))

        struct_expr = pl.struct(close_expr.alias("_c"), benchmark_expr.alias("_b"))
        beta_expr = struct_expr.map_batches(compute_beta, return_dtype=pl.Float64)
    else:
        # Returns: r = price / price[1] - 1
        close_ret = close_expr / close_expr.shift(1) - 1.0
        bench_ret = benchmark_expr / benchmark_expr.shift(1) - 1.0

        # Population covariance / variance over `length` return observations.
        # ddof cancels in the ratio, so use ddof=0 for both.
        cov = pl.rolling_cov(close_ret, bench_ret, window_size=length, min_samples=length, ddof=0)
        var = close_ret.rolling_var(window_size=length, min_samples=length, ddof=0)
        beta_expr = cov / var

    if offset != 0:
        beta_expr = beta_expr.shift(offset)

    return beta_expr.alias(f"BETA_{length}")
