# -*- coding: utf-8 -*-
# =============================================================================
# Polars Keltner Channels Implementation (Pure Composition)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.utils import v_pos_int
from polars_ti.utils._validate import v_expr


def kc(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 20,
    scalar: float = 2.0,
    mamode: str = "ema",
    tr: bool = True,
    talib: bool = True,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Keltner Channels (KC)

    Pure composition: pl_ma for basis/band, pl_true_range for TR.

    A popular volatility indicator similar to Bollinger Bands.

    Sources:
        https://www.tradingview.com/wiki/Keltner_Channels_(KC)

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        length: The period. Default: 20
        scalar: Band multiplier. Default: 2.0
        mamode: MA type for basis/band. Default: 'ema'
        tr: Use True Range (True) or High-Low (False). Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with kcl (Lower), kcb (Basis), kcu (Upper) columns
    """
    from polars_ti.ma import ma
    from polars_ti.volatility.true_range import true_range

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    length = v_pos_int(length, "length")

    # Range: True Range or High-Low (matches Pandas: true_range() if tr else high_low_range())
    if tr:
        range_expr = true_range(high_expr, low_expr, close_expr, talib=talib)
    else:
        range_expr = high_expr - low_expr

    # Basis = MA(close) and Band = MA(range) using pl_ma composition.
    # OLD kc never propagated talib to its MAs, so honour talib (default True)
    # to match the OLD golden's TA-Lib MA in talib mode.
    basis = ma(name=mamode, source=close_expr, length=length, talib=talib)
    band = ma(name=mamode, source=range_expr, length=length, talib=talib)

    # KC bands
    lower = basis - pl.lit(scalar) * band
    upper = basis + pl.lit(scalar) * band

    # Apply offset
    if offset != 0:
        lower = lower.shift(offset)
        basis = basis.shift(offset)
        upper = upper.shift(offset)

    _props = f"_{mamode.lower()[0] if mamode else 'e'}_{length}_{int(scalar)}"

    return pl.struct(
        [
            lower.alias("kcl"),
            basis.alias("kcb"),
            upper.alias("kcu"),
        ]
    ).alias(f"KC{_props}")
