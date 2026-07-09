# -*- coding: utf-8 -*-
# =============================================================================
# Polars AOBV (Archer On Balance Volume) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr, v_pos_int


def aobv(
    close: IntoExpr,
    volume: IntoExpr,
    fast: int = 4,
    slow: int = 12,
    max_lookback: int = 2,
    min_lookback: int = 2,
    mamode: str = "ema",
    run_length: int = 2,
    talib: bool = True,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Archer On Balance Volume (AOBV)

    Archer On Balance Volume provides additional indicator analysis on OBV.
    It calculates moving averages of OBV as well as Long and Short Run Trends.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        fast: Fast MA period. Default: 4
        slow: Slow MA period. Default: 12
        max_lookback: Maximum OBV lookback. Default: 2
        min_lookback: Minimum OBV lookback. Default: 2
        mamode: MA type. Default: 'ema'
        run_length: Long/Short run length. Default: 2
        talib: If True and TA-Lib installed, use TA-Lib for OBV. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: List of expressions [OBV, OBV_min, OBV_max, OBV_fast, OBV_slow, LR, SR]
    """
    from polars_ti.volume.obv import obv
    from polars_ti.ma import ma
    from polars_ti.trend.long_run import long_run
    from polars_ti.trend.short_run import short_run

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    fast = v_pos_int(fast, "fast")
    slow = v_pos_int(slow, "slow")

    if slow < fast:
        fast, slow = slow, fast

    _mode = mamode.lower()[0] if len(mamode) else ""

    # Build OBV base
    obv_expr = obv(close_expr, volume_expr, talib=talib, offset=0)

    # Build MA expressions on OBV
    obv_fast_ma = ma(name=mamode, source=obv_expr, length=fast, talib=talib)
    obv_slow_ma = ma(name=mamode, source=obv_expr, length=slow, talib=talib)

    # Long/Short run on the MAs
    obv_long = long_run(obv_fast_ma, obv_slow_ma, length=run_length)
    obv_short = short_run(obv_fast_ma, obv_slow_ma, length=run_length)

    # Rolling min/max
    obv_min = obv_expr.rolling_min(window_size=min_lookback, min_samples=min_lookback)
    obv_max = obv_expr.rolling_max(window_size=max_lookback, min_samples=max_lookback)

    # Apply offset
    if offset != 0:
        obv_expr = obv_expr.shift(offset)
        obv_min = obv_min.shift(offset)
        obv_max = obv_max.shift(offset)
        obv_fast_ma = obv_fast_ma.shift(offset)
        obv_slow_ma = obv_slow_ma.shift(offset)
        obv_long = obv_long.shift(offset)
        obv_short = obv_short.shift(offset)

    # Rename with proper aliases
    return [
        obv_expr.alias("OBV"),
        obv_min.alias(f"OBV_min_{min_lookback}"),
        obv_max.alias(f"OBV_max_{max_lookback}"),
        obv_fast_ma.alias(f"OBV{_mode}_{fast}"),
        obv_slow_ma.alias(f"OBV{_mode}_{slow}"),
        obv_long.alias(f"AOBV_LR_{run_length}"),
        obv_short.alias(f"AOBV_SR_{run_length}"),
    ]
