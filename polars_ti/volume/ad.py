# -*- coding: utf-8 -*-
# =============================================================================
# Polars AD (Accumulation/Distribution) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._math import pl_non_zero_range
from polars_ti.utils._validate import v_expr


def pl_ad(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Accumulation/Distribution (AD)

    Accumulation/Distribution indicator utilizes the relative position
    of the close to its High-Low range with volume then accumulated.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: AD expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if any(e is None for e in [high_expr, low_expr, close_expr, volume_expr]):
        return None

    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:

        def compute_ad(df: pl.DataFrame) -> pl.Series:
            from talib import AD as TALIB_AD

            h = df["high"].to_numpy().astype(np.float64)
            l = df["low"].to_numpy().astype(np.float64)
            c = df["close"].to_numpy().astype(np.float64)
            v = df["volume"].to_numpy().astype(np.float64)
            result = TALIB_AD(h, l, c, v)
            return pl.Series("AD", result)

        # Need struct to pass multiple columns
        ad_expr = pl.struct(
            [
                high_expr.alias("high"),
                low_expr.alias("low"),
                close_expr.alias("close"),
                volume_expr.alias("volume"),
            ]
        ).map_batches(lambda s: compute_ad(s.struct.unnest()), return_dtype=pl.Float64)
    else:
        # Pure Polars: AD = cumsum(volume * (2*close - high - low) / (high - low))
        hl_range_safe = pl_non_zero_range(high_expr, low_expr)
        clv = (2 * close_expr - high_expr - low_expr) / hl_range_safe
        ad_expr = (clv * volume_expr).cum_sum()

    if offset != 0:
        ad_expr = ad_expr.shift(offset)

    return ad_expr.alias("AD")
