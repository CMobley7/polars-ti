# -*- coding: utf-8 -*-
# =============================================================================
# Polars NATR Implementation (Pure Composition)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr


def natr(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    scalar: float = 100.0,
    mamode: str = "rma",
    talib: bool = False,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Normalized Average True Range (NATR)

    Pure composition: pl_atr / close * scalar.

    NATR normalizes ATR by dividing by close price.

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/normalized-average-true-range-natr/

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        length: ATR period. Default: 14
        scalar: Magnification factor. Default: 100.0
        mamode: MA type for ATR. Default: 'rma' (Wilder; aligns with ATR and TA-Lib)
        talib: Use TA-Lib compatible ATR. Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: NATR expression
    """
    import numpy as np

    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    from polars_ti.volatility.atr import atr

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    if Imports["talib"] and v_talib(talib) and scalar == 100.0 and mamode == "rma":
        # Fast path: TA-Lib's dedicated NATR is a single C call (NATR is a fixed
        # 100*ATR/close, so only the scalar=100 default routes here).
        def _compute(s: pl.Series) -> pl.Series:
            from talib import NATR as _NATR

            d = s.struct.unnest()
            return pl.Series(
                _NATR(
                    d["_h"].to_numpy().astype(np.float64),
                    d["_l"].to_numpy().astype(np.float64),
                    d["_c"].to_numpy().astype(np.float64),
                    timeperiod=length,
                )
            )

        result = pl.struct(high_expr.alias("_h"), low_expr.alias("_l"), close_expr.alias("_c")).map_batches(
            _compute, return_dtype=pl.Float64
        )
    else:
        # NATR = (scalar / close) * ATR (native atr now matches talib.ATR too)
        atr_result = atr(high_expr, low_expr, close_expr, length=length, mamode=mamode, talib=talib)
        result = (pl.lit(scalar) / close_expr) * atr_result

    # Apply offset
    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"NATR_{length}")
