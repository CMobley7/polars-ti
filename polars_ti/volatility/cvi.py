# -*- coding: utf-8 -*-
# =============================================================================
# Polars CVI (Chaikin Volatility) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr, v_pos_int


def cvi(
    high: IntoExpr,
    low: IntoExpr,
    length: int = 10,
    offset: int = 0,
) -> PlExpr:
    """Polars: Chaikins Volatility (CVI)

    Measures the range between the High and Low prices by calculating
    the rate of change of the EMA of the High-Low spread.

    Formula:
        HL    = High - Low
        EMAHL = EMA(HL, length)
        CVI   = 100 * (EMAHL - EMAHL[length]) / EMAHL[length]

    Sources:
        Marc Chaikin
        https://school.stockcharts.com/doku.php?id=technical_indicators:chaikins_volatility

    Args:
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        length: EMA period and lookback. Default: 10
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: CVI expression.
    """
    from polars_ti.overlap.ema import ema

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    if high_expr is None or low_expr is None:
        return None

    _length = v_pos_int(length, "length")
    _offset = offset

    hl_expr = high_expr - low_expr
    ema_hl = ema(hl_expr, length=_length, talib=False)

    result = pl.lit(100.0) * (ema_hl - ema_hl.shift(_length)) / ema_hl.shift(_length)

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias(f"CVI_{length}")
