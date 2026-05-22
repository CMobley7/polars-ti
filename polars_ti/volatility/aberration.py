# -*- coding: utf-8 -*-
# =============================================================================
# Polars ABERRATION Implementation (Composition: pl_atr + pl_sma)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_aberration(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 5,
    atr_length: int = 15,
    talib: bool = True,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Aberration (ABER)

    A volatility indicator similar to Keltner Channels.
    Returns a struct with ZG, SG, XG, ATR columns.

    Uses composition: pl_hlc3, pl_sma, and pl_atr.

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        length: SMA period. Default: 5
        atr_length: ATR period. Default: 15
        talib: If True and TA-Lib installed, uses TA-Lib for ATR. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with ZG, SG, XG, ATR columns
    """
    from polars_ti.volatility.atr import pl_atr
    from polars_ti.overlap.hlc3 import pl_hlc3
    from polars_ti.overlap.sma import pl_sma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    # HLC3 using pl_hlc3 composition
    hlc3_expr = pl_hlc3(high_expr, low_expr, close_expr)

    # ZG = SMA(HLC3, length) using pl_sma composition
    zg = pl_sma(hlc3_expr, length=length)

    # ATR using pl_atr composition
    atr_expr = pl_atr(high_expr, low_expr, close_expr, length=atr_length, talib=talib)

    # SG = ZG + ATR, XG = ZG - ATR
    sg = zg + atr_expr
    xg = zg - atr_expr

    # Apply offset
    if offset != 0:
        zg = zg.shift(offset)
        sg = sg.shift(offset)
        xg = xg.shift(offset)
        atr_expr = atr_expr.shift(offset)

    _props = f"_{length}_{atr_length}"

    return pl.struct(
        [
            zg.alias(f"ABER_ZG{_props}"),
            sg.alias(f"ABER_SG{_props}"),
            xg.alias(f"ABER_XG{_props}"),
            atr_expr.alias(f"ABER_ATR{_props}"),
        ]
    ).alias(f"ABER{_props}")
