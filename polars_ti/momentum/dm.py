# -*- coding: utf-8 -*-
# =============================================================================
# Polars DM (Directional Movement) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_dm(
    high: IntoExpr,
    low: IntoExpr,
    length: int = 14,
    mamode: str = "rma",
    talib: bool = True,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Directional Movement (DM)

    Compares prior highs and lows to yield +DM and -DM series.
    Developed by J. Welles Wilder in 1978.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        length: Period. Default: 14
        mamode: MA type. Default: 'rma'
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result. Default: 0

    Returns:
        list[pl.Expr]: [DMP_14, DMN_14] expressions
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib, v_mamode
    from polars_ti.ma import pl_ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    _use_talib = Imports["talib"] and v_talib(talib)
    _mamode = v_mamode(mamode, "rma")

    if _use_talib:
        _length = length

        def compute_dmp(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            high_arr = df["high"].to_numpy().astype(np.float64)
            low_arr = df["low"].to_numpy().astype(np.float64)
            from talib import PLUS_DM

            result = PLUS_DM(high_arr, low_arr, timeperiod=_length)
            return pl.Series(result)

        def compute_dmn(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            high_arr = df["high"].to_numpy().astype(np.float64)
            low_arr = df["low"].to_numpy().astype(np.float64)
            from talib import MINUS_DM

            result = MINUS_DM(high_arr, low_arr, timeperiod=_length)
            return pl.Series(result)

        struct_expr = pl.struct(high=high_expr, low=low_expr)
        dmp_expr = struct_expr.map_batches(compute_dmp, return_dtype=pl.Float64)
        dmn_expr = struct_expr.map_batches(compute_dmn, return_dtype=pl.Float64)
    else:
        # Pure Polars: up = high - high.shift(1), dn = low.shift(1) - low
        up = high_expr - high_expr.shift(1)
        dn = low_expr.shift(1) - low_expr

        # +DM: when up > dn AND up > 0
        pos_raw = pl.when((up > dn) & (up > 0)).then(up).otherwise(0.0)
        # -DM: when dn > up AND dn > 0
        neg_raw = pl.when((dn > up) & (dn > 0)).then(dn).otherwise(0.0)

        # Smooth with MA
        dmp_expr = pl_ma(_mamode, pos_raw, length=length, offset=0)
        dmn_expr = pl_ma(_mamode, neg_raw, length=length, offset=0)

    if offset != 0:
        dmp_expr = dmp_expr.shift(offset)
        dmn_expr = dmn_expr.shift(offset)

    return [dmp_expr.alias(f"DMP_{length}"), dmn_expr.alias(f"DMN_{length}")]
