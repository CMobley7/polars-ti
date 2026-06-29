# -*- coding: utf-8 -*-
# =============================================================================
# Polars CKSP Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def cksp(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    p: int = 10,
    x: float | None = None,
    q: int | None = None,
    tvmode: bool | None = None,
    mamode: str | None = None,
    talib: bool = False,
    offset: int = 0,
) -> PlExpr:
    """Polars: Chande Kroll Stop (CKSP)

    Trend-following indicator using ATR-based stops. Defaults to the original
    book implementation (``tvmode=None``/``False``): ``p=10, x=3, q=20`` with a
    simple moving average for the ATR. Pass ``tvmode=True`` for the TradingView
    variant (``p=10, x=1, q=9`` with Wilder's RMA).

    Defaults:
        Book:         p=10, x=3, q=20, ma=sma
        Trading View: p=10, x=1, q=9,  ma=rma

    Formula:
        long_stop = rolling_max(high, p) - x * ATR(p)  -> rolling_max(q)
        short_stop = rolling_min(low, p) + x * ATR(p)  -> rolling_min(q)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        p: ATR and first stop period. Default: 10
        x: ATR multiplier. Default: 1 (TV mode) / 3 (book mode)
        q: Second stop period. Default: 9 (TV mode) / 20 (book mode)
        tvmode: TradingView (True) or book (None/False) implementation.
            Default: None (book)
        mamode: MA mode for ATR. Default: 'rma' (TV) / 'sma' (book)
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with CKSPl and CKSPs columns
    """
    from polars_ti.volatility.atr import atr

    mode_tv = tvmode is True

    # x / q defaults depend on mode (matches the old pandas implementation)
    x = float(x) if isinstance(x, (int, float)) and x > 0 else (1 if mode_tv else 3)
    q = int(q) if isinstance(q, (int, float)) and q > 0 else (9 if mode_tv else 20)

    if mamode is None:
        mamode = "rma" if mode_tv else "sma"

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    atr_expr = atr(high_expr, low_expr, close_expr, length=p, mamode=mamode, talib=talib)

    long_stop_ = high_expr.rolling_max(window_size=p) - x * atr_expr
    long_stop = long_stop_.rolling_max(window_size=q)

    short_stop_ = low_expr.rolling_min(window_size=p) + x * atr_expr
    short_stop = short_stop_.rolling_min(window_size=q)

    if offset != 0:
        long_stop = long_stop.shift(offset)
        short_stop = short_stop.shift(offset)

    _props = f"_{p}_{x}_{q}"
    return pl.struct(
        long_stop.alias(f"CKSPl{_props}"),
        short_stop.alias(f"CKSPs{_props}"),
    ).alias(f"CKSP{_props}")
