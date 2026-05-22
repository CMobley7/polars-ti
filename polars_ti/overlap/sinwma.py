# -*- coding: utf-8 -*-
# =============================================================================
# Polars SINWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_sinwma(
    close: IntoExpr,
    length: int = 14,
    offset: int = 0,
) -> PlExpr:
    """Polars: Sine Weighted Moving Average (SINWMA)

    A weighted average using sine cycles. The middle term(s) of the average
    have the highest weight(s).

    Source:
        https://www.tradingview.com/script/6MWFvnPO-Sine-Weighted-Moving-Average/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 14
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: SINWMA expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Calculate sine weights
    import numpy as np

    sine_weights = np.array([np.sin((i + 1) * np.pi / (length + 1)) for i in range(length)])
    weights_sum = sine_weights.sum()
    weights_list = (sine_weights / weights_sum).tolist()

    _length = length
    _weights = weights_list

    def sine_weighted_mean(s: pl.Series) -> float:
        vals = s.to_numpy()
        if len(vals) < _length:
            return float("nan")
        return (vals * _weights[-len(vals) :]).sum()

    sinwma_expr = close_expr.rolling_map(function=sine_weighted_mean, window_size=length, min_samples=length)

    # Apply offset
    if offset != 0:
        sinwma_expr = sinwma_expr.shift(offset)

    return sinwma_expr.alias(f"SINWMA_{length}")
