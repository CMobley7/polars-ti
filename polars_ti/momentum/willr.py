# -*- coding: utf-8 -*-
# =============================================================================
# Polars WILLR (Williams %R) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_willr(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Williams %R (WILLR)

    Momentum oscillator for overbought/oversold conditions.
    Ranges from -100 to 0, where -80 to -100 is oversold and -20 to 0 is overbought.

    Sources:
        https://www.investopedia.com/terms/w/williamsr.asp
        https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r

    Calculation:
        Highest High = rolling_max(high, length)
        Lowest Low = rolling_min(low, length)
        WILLR = 100 * ((close - Lowest Low) / (Highest High - Lowest Low) - 1)
             = -100 * (Highest High - close) / (Highest High - Lowest Low)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Lookback period. Default: 14
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: WILLR expression (-100 to 0 range)
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib as validate_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    _use_talib = Imports["talib"] and validate_talib(talib)

    if _use_talib:
        _length = length

        def compute_willr(df: pl.DataFrame) -> pl.Series:
            from talib import WILLR

            h = df["high"].to_numpy().astype(np.float64)
            l = df["low"].to_numpy().astype(np.float64)
            c = df["close"].to_numpy().astype(np.float64)
            result = WILLR(h, l, c, _length)
            return pl.Series(f"WILLR_{_length}", result)

        willr_expr = pl.struct(
            [
                high_expr.alias("high"),
                low_expr.alias("low"),
                close_expr.alias("close"),
            ]
        ).map_batches(lambda s: compute_willr(s.struct.unnest()), return_dtype=pl.Float64)
    else:
        # Native Polars implementation
        lowest_low = low_expr.rolling_min(window_size=length, min_samples=length)
        highest_high = high_expr.rolling_max(window_size=length, min_samples=length)

        willr_expr = 100 * ((close_expr - lowest_low) / (highest_high - lowest_low) - 1)

    if offset != 0:
        willr_expr = willr_expr.shift(offset)

    return willr_expr.alias(f"WILLR_{length}")
