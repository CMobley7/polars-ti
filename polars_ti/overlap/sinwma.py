# -*- coding: utf-8 -*-
# =============================================================================
# Polars SINWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def sinwma(
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

    import numpy as np

    _length = length
    _weights: list[float] | None = None

    def sine_weighted_mean(s: pl.Series) -> float:
        nonlocal _weights
        vals = s.to_numpy()
        if len(vals) < _length:
            return float("nan")
        if _weights is None:
            # Build the length-sized weight vector lazily — only once a full window
            # exists — so an absurd length (>> data) returns all-null instead of
            # eagerly allocating an O(length) array (a hang/OOM on e.g. length=1e9).
            sine_weights = np.array([np.sin((i + 1) * np.pi / (_length + 1)) for i in range(_length)])
            _weights = (sine_weights / sine_weights.sum()).tolist()
        return (vals * _weights[-len(vals) :]).sum()

    sinwma_expr = close_expr.rolling_map(function=sine_weighted_mean, window_size=length, min_samples=length)

    # Apply offset
    if offset != 0:
        sinwma_expr = sinwma_expr.shift(offset)

    return sinwma_expr.alias(f"SINWMA_{length}")
