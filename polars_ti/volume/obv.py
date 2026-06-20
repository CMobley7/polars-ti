# -*- coding: utf-8 -*-
# =============================================================================
# Polars OBV (On Balance Volume) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def obv(
    close: IntoExpr,
    volume: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: On Balance Volume (OBV)

    On Balance Volume is a cumulative indicator to measure buying and selling
    pressure.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: OBV expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:

        def compute_obv(df: pl.DataFrame) -> pl.Series:
            from talib import OBV as TALIB_OBV

            c = df["close"].to_numpy().astype(np.float64)
            v = df["volume"].to_numpy().astype(np.float64)
            result = TALIB_OBV(c, v)
            return pl.Series("OBV", result)

        obv_expr = pl.struct([close_expr.alias("close"), volume_expr.alias("volume")]).map_batches(
            lambda s: compute_obv(s.struct.unnest()), return_dtype=pl.Float64
        )
    else:
        # Pure Polars: OBV = cumsum(signed_volume)
        # signed_volume = volume * sign(close.diff())
        close_diff = close_expr.diff()
        sign = pl.when(close_diff > 0).then(1).when(close_diff < 0).then(-1).otherwise(0)
        # First value should use 1 as initial sign (matching pandas-ta)
        sign = sign.fill_null(1)
        obv_expr = (sign * volume_expr).cum_sum()

    if offset != 0:
        obv_expr = obv_expr.shift(offset)

    return obv_expr.alias("OBV")
