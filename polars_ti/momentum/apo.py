# -*- coding: utf-8 -*-
# =============================================================================
# Polars APO (Absolute Price Oscillator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_apo(
    close: IntoExpr,
    fast: int = 12,
    slow: int = 26,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Absolute Price Oscillator (APO)

    Measures momentum using the difference between fast and slow MAs.
    APO = MA(close, fast) - MA(close, slow)

    Args:
        close: Column name or pl.Expr for 'close' prices
        fast: Short period. Default: 12
        slow: Long period. Default: 26
        mamode: Moving average type. Default: 'sma'
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: APO expression
    """
    import numpy as np
    from polars_ti.maps import Imports
    from polars_ti.ma import pl_ma
    from polars_ti.utils import v_talib, tal_ma

    if slow < fast:
        fast, slow = slow, fast

    close_expr = v_expr(close)
    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:
        _fast = fast
        _slow = slow
        _mamode = mamode

        def compute_apo(s: pl.Series) -> pl.Series:
            from talib import APO as TALIB_APO

            arr = s.to_numpy().astype(np.float64)
            result = TALIB_APO(arr, _fast, _slow, tal_ma(_mamode))
            return pl.Series(f"APO_{_fast}_{_slow}", result)

        apo_expr = close_expr.map_batches(compute_apo, return_dtype=pl.Float64)
    else:
        # Use pl_ma for code reuse
        fast_ma = pl_ma(name=mamode, source=close_expr, length=fast, talib=False)
        slow_ma = pl_ma(name=mamode, source=close_expr, length=slow, talib=False)
        apo_expr = fast_ma - slow_ma

    if offset != 0:
        apo_expr = apo_expr.shift(offset)

    return apo_expr.alias(f"APO_{fast}_{slow}")
