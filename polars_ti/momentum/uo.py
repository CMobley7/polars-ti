# -*- coding: utf-8 -*-
# =============================================================================
# Polars UO (Ultimate Oscillator) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_uo(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    fast: int = 7,
    medium: int = 14,
    slow: int = 28,
    fast_w: float = 4.0,
    medium_w: float = 2.0,
    slow_w: float = 1.0,
    talib: bool = True,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Ultimate Oscillator (UO)

    The Ultimate Oscillator is a momentum indicator over three different
    periods. It attempts to correct false divergence trading signals.

    Sources:
        https://www.investopedia.com/terms/u/ultimateoscillator.asp
        https://school.stockcharts.com/doku.php?id=technical_indicators:ultimate_oscillator

    Calculation:
        BP = Close - min(Low, PreviousClose)
        TR = max(High, PreviousClose) - min(Low, PreviousClose)
        Fast_avg = sum(BP, fast) / sum(TR, fast)
        Medium_avg = sum(BP, medium) / sum(TR, medium)
        Slow_avg = sum(BP, slow) / sum(TR, slow)
        UO = 100 * (fast_w * Fast + medium_w * Medium + slow_w * Slow) / total_weight

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        fast: Fast period. Default: 7
        medium: Medium period. Default: 14
        slow: Slow period. Default: 28
        fast_w: Fast weight. Default: 4.0
        medium_w: Medium weight. Default: 2.0
        slow_w: Slow weight. Default: 1.0
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        drift: Periods for previous close. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Ultimate Oscillator values
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib as validate_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    _props = f"_{fast}_{medium}_{slow}"
    _use_talib = Imports["talib"] and validate_talib(talib)

    if _use_talib:
        _fast = fast
        _medium = medium
        _slow = slow

        def compute_uo(df: pl.DataFrame) -> pl.Series:
            from talib import ULTOSC
            h = df["high"].to_numpy().astype(np.float64)
            l = df["low"].to_numpy().astype(np.float64)
            c = df["close"].to_numpy().astype(np.float64)
            result = ULTOSC(h, l, c, _fast, _medium, _slow)
            return pl.Series(f"UO{_props}", result)

        # Build struct and compute
        uo_expr = (
            pl.struct([
                high_expr.alias("high"),
                low_expr.alias("low"),
                close_expr.alias("close"),
            ])
            .map_batches(
                lambda s: compute_uo(s.struct.unnest()),
                return_dtype=pl.Float64
            )
        )
    else:
        # Native Polars implementation
        # BP = Close - min(Low, PreviousClose)
        prev_close = close_expr.shift(drift)
        min_l_or_pc = pl.min_horizontal(low_expr, prev_close)
        bp = close_expr - min_l_or_pc

        # TR = max(High, PreviousClose) - min(Low, PreviousClose)
        max_h_or_pc = pl.max_horizontal(high_expr, prev_close)
        tr = max_h_or_pc - min_l_or_pc

        # Rolling sums for each period
        fast_bp = bp.rolling_sum(window_size=fast, min_samples=fast)
        fast_tr = tr.rolling_sum(window_size=fast, min_samples=fast)
        fast_avg = fast_bp / fast_tr

        medium_bp = bp.rolling_sum(window_size=medium, min_samples=medium)
        medium_tr = tr.rolling_sum(window_size=medium, min_samples=medium)
        medium_avg = medium_bp / medium_tr

        slow_bp = bp.rolling_sum(window_size=slow, min_samples=slow)
        slow_tr = tr.rolling_sum(window_size=slow, min_samples=slow)
        slow_avg = slow_bp / slow_tr

        # UO = 100 * weighted_sum / total_weight
        total_weight = fast_w + medium_w + slow_w
        weighted_sum = (fast_w * fast_avg) + (medium_w * medium_avg) + (slow_w * slow_avg)
        uo_expr = 100 * weighted_sum / total_weight

    if offset != 0:
        uo_expr = uo_expr.shift(offset)

    return uo_expr.alias(f"UO{_props}")
