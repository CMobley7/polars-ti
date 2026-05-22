# -*- coding: utf-8 -*-
from numpy import isnan, nan, uintc, zeros_like
from numba import njit


@njit(cache=True)
def nb_atrts(x, ma, atr_, length, ma_length):
    m = x.size
    k = max(length, ma_length)

    result = x.copy()
    up = zeros_like(x, dtype=uintc)
    dn = zeros_like(x, dtype=uintc)

    expn = x > ma
    up[expn], dn[~expn] = 1, 1
    up[:k], dn[:k] = 0, 0
    result[:k] = nan

    for i in range(k, m):
        pr = result[i - 1]
        if up[i]:
            result[i] = x[i] - atr_[i]
            if result[i] < pr:
                result[i] = pr
        if dn[i]:
            result[i] = x[i] + atr_[i]
            if result[i] > pr:
                result[i] = pr

    long, short = result * up, result * dn
    long[long == 0], short[short == 0] = nan, nan

    return result, long, short


# =============================================================================
# Polars ATRTS Implementation (Composition: pl_atr + pl_ma + nb_atrts kernel)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_atrts(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    ma_length: int = 20,
    multiplier: float = 3.0,
    mamode: str = "ema",
    offset: int = 0,
) -> pl.Expr:
    """Polars: ATR Trailing Stop (ATRTS)

    Uses composition: pl_atr + pl_ma for ATR and MA calculation,
    then applies the nb_atrts kernel for the trailing stop logic.

    Sources:
        https://www.motivewave.com/studies/atr_trailing_stops.htm

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        length: ATR length. Default: 14
        ma_length: MA length. Default: 20
        multiplier: ATR multiplier. Default: 3.0
        mamode: MA type ('ema', 'sma', 'rma'). Default: 'ema'
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ATRTS expression
    """
    from polars_ti.volatility.atr import pl_atr
    from polars_ti.ma import pl_ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    _length = length
    _ma_length = ma_length
    _multiplier = multiplier
    _mamode = mamode.lower()
    _offset = offset

    def compute_atrts(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        close_arr = df["_close"].to_numpy().astype(np.float64)
        atr_arr = df["_atr"].to_numpy().astype(np.float64) * _multiplier
        ma_arr = df["_ma"].to_numpy().astype(np.float64)

        # Call the nb_atrts kernel (from Pandas section - line 19)
        result, _, _ = nb_atrts(close_arr, ma_arr, atr_arr, _length, _ma_length)

        if _offset != 0:
            result = np.roll(result, _offset)
            if _offset > 0:
                result[:_offset] = np.nan

        return pl.Series(result)

    # Use composition: pl_atr for ATR, pl_ma for MA (just like Pandas!)
    atr_expr = pl_atr(high_expr, low_expr, close_expr, length=length, mamode="ema", talib=False)
    ma_expr = pl_ma(name=_mamode, source=close_expr, length=ma_length, talib=False)

    return (
        pl.struct(
            [
                close_expr.alias("_close"),
                atr_expr.alias("_atr"),
                ma_expr.alias("_ma"),
            ]
        )
        .map_batches(compute_atrts, return_dtype=pl.Float64)
        .alias(f"ATRTS{mamode[0]}_{length}_{ma_length}_{multiplier}")
    )
