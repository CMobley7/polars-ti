# -*- coding: utf-8 -*-
# =============================================================================
# Polars AMAT Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def amat(
    close: IntoExpr,
    fast: int = 8,
    slow: int = 21,
    lookback: int = 2,
    mamode: str = "ema",
    offset: int = 0,
) -> PlExpr:
    """Polars: Archer Moving Averages Trends (AMAT)

    Creates long run and short run trend signals from fast/slow MA crossovers.

    Args:
        close: Column name or pl.Expr for input values
        fast: Fast MA period. Default: 8
        slow: Slow MA period. Default: 21
        lookback: Lookback for long_run/short_run. Default: 2
        mamode: MA type. Default: 'ema'
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with AMAT_LR and AMAT_SR columns
    """
    from polars_ti.ma import ma
    from polars_ti.trend.long_run import long_run
    from polars_ti.trend.short_run import short_run
    from polars_ti.utils import v_pos_int

    close_expr = v_expr(close)

    fast = v_pos_int(fast, "fast")
    slow = v_pos_int(slow, "slow")

    fast_ma = ma(mamode, close_expr, length=fast, talib=False)
    slow_ma = ma(mamode, close_expr, length=slow, talib=False)

    lr = long_run(fast_ma, slow_ma, length=lookback)
    sr = short_run(fast_ma, slow_ma, length=lookback)

    if offset != 0:
        lr = lr.shift(offset)
        sr = sr.shift(offset)

    _props = f"_{fast}_{slow}_{lookback}"
    return pl.struct(
        lr.alias(f"AMAT{mamode[0]}_LR{_props}"),
        sr.alias(f"AMAT{mamode[0]}_SR{_props}"),
    ).alias(f"AMAT{mamode[0]}{_props}")
