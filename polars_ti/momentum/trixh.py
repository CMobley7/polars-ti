# -*- coding: utf-8 -*-
# =============================================================================
# Polars TRIXH Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def trixh(
    close: IntoExpr,
    length: int = 18,
    signal: int = 9,
    scalar: float = 100.0,
    drift: int = 1,
    talib: bool = True,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: TRIX Histogram (TRIXH)

    TRIX Histogram extends the TRIX indicator by adding a histogram that
    represents the difference between TRIX and its signal line, similar to
    MACD histogram, helping identify momentum changes and divergences.

    Sources:
        https://www.investopedia.com/terms/t/trix.asp
        https://school.stockcharts.com/doku.php?id=technical_indicators:trix

    Calculation:
        TRIX = TRIX(close, length, scalar, drift)
        Signal = SMA(TRIX, signal)
        Histogram = TRIX - Signal

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: TRIX period (EMA applied 3 times). Default: 18
        signal: Signal SMA period. Default: 9
        scalar: Multiplication factor. Default: 100
        drift: Periods for pct_change. Default: 1
        talib: If True and TA-Lib installed, use TA-Lib for TRIX. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: [TRIX, TRIX_signal, TRIX_histogram]
    """
    from polars_ti.momentum.trix import trix

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Swap if length < signal (matching pandas behavior)
    if length < signal:
        length, signal = signal, length

    _props = f"_{length}_{signal}"

    # Delegate the TRIX line and its signal to trix() (single source of truth,
    # mirroring the pandas baseline). trix() returns a struct exposing the TRIX
    # and TRIXs fields for both the TA-Lib and native paths; the histogram is
    # simply their difference. This avoids reimplementing — and silently
    # diverging from — trix()'s triple-EMA warm-up seeding on the native path.
    trix_struct = trix(
        close_expr,
        length=length,
        signal=signal,
        scalar=scalar,
        drift=drift,
        talib=talib,
        offset=0,  # applied below to all three outputs
    )

    if trix_struct is None:
        return None

    trix_expr = trix_struct.struct.field(f"TRIX{_props}")
    trix_signal_expr = trix_struct.struct.field(f"TRIXs{_props}")

    # Histogram = TRIX - Signal
    histogram_expr = trix_expr - trix_signal_expr

    if offset != 0:
        trix_expr = trix_expr.shift(offset)
        trix_signal_expr = trix_signal_expr.shift(offset)
        histogram_expr = histogram_expr.shift(offset)

    return [
        trix_expr.alias(f"TRIX{_props}"),
        trix_signal_expr.alias(f"TRIXs{_props}"),
        histogram_expr.alias(f"TRIXh{_props}"),
    ]
