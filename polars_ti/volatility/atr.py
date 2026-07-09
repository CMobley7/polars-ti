# -*- coding: utf-8 -*-
# =============================================================================
# Polars ATR Implementation (Simple Composition: pl_true_range + pl_ma)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.maps import Imports
from polars_ti.utils._validate import v_expr
from polars_ti.utils import v_talib


def atr(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    mamode: str = "rma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Average True Range (ATR)

    ATR = MA(TrueRange)

    Uses TA-Lib when available and talib=True, otherwise composes
    pl_true_range + pl_ma - exactly like the Pandas version.

    Sources:
        https://www.tradingview.com/wiki/Average_True_Range_(ATR)
        https://www.investopedia.com/terms/a/atr.asp

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        length: ATR period. Default: 14
        mamode: MA type ('rma', 'sma', 'ema', etc.). Default: "rma"
        talib: If True and TA-Lib is installed, uses TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ATR expression
    """
    from polars_ti.volatility.true_range import true_range
    from polars_ti.ma import ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib)
    _mamode = mamode.lower() if isinstance(mamode, str) else "rma"

    if _use_talib:
        # TA-Lib path
        _length = length
        _offset = offset

        def compute_atr_talib(struct: pl.Series) -> pl.Series:
            from talib import ATR as TALIB_ATR

            df = struct.struct.unnest()
            result = TALIB_ATR(
                df["_high"].to_numpy().astype(np.float64),
                df["_low"].to_numpy().astype(np.float64),
                df["_close"].to_numpy().astype(np.float64),
                timeperiod=_length,
            )
            if _offset != 0:
                result = np.roll(result, _offset)
                if _offset > 0:
                    result[:_offset] = np.nan
                else:
                    result[_offset:] = np.nan
            return pl.Series(result)

        return (
            pl.struct(
                [
                    high_expr.alias("_high"),
                    low_expr.alias("_low"),
                    close_expr.alias("_close"),
                ]
            )
            .map_batches(compute_atr_talib, return_dtype=pl.Float64)
            .alias(f"ATR{_mamode[0]}_{length}")
        )
    else:
        # Simple composition: TR → MA (just like Pandas!)
        # TA-Lib's ATR excludes the first bar's true range (no prior close), then
        # seeds the Wilder average over TR[1..length] at index ``length``. The
        # native RMA seed is now leading-NaN-tolerant (mirrors _ema_numba), so we
        # null the first TR bar to reproduce that warmup exactly — native ATR then
        # equals talib.ATR to float noise post-warmup. (The null no longer poisons
        # the MA: the seed re-anchors at the first finite TR.)
        tr_expr = true_range(high_expr, low_expr, close_expr, talib=False)
        if _mamode == "rma":
            tr_expr = pl.when(close_expr.shift(1).is_null()).then(None).otherwise(tr_expr)
        atr_expr = ma(name=_mamode, source=tr_expr, length=length, talib=False, presma=True)

        if offset != 0:
            atr_expr = atr_expr.shift(offset)

        return atr_expr.alias(f"ATR{_mamode[0]}_{length}")
