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

    # Get TRIX and Signal from pl_trix
    # Note: pl_trix default length is 30, we use 18 for trixh
    trix_exprs = trix(
        close_expr,
        length=length,
        signal=signal,
        scalar=scalar,
        drift=drift,
        talib=talib,
        offset=0,  # Apply offset at the end
    )

    if trix_exprs is None:
        return None

    # Extract the TRIX and Signal expressions (before alias)
    # We need to compute histogram = trix - signal
    # But since pl_trix returns aliased expressions, we need to rebuild

    # Rebuild TRIX calculation to get the raw expressions for histogram computation
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    from polars_ti.overlap.ema import ema
    import numpy as np

    _use_talib = Imports["talib"] and v_talib(talib)
    _length = length

    if _use_talib:

        def compute_trix(s: pl.Series) -> pl.Series:
            from talib import TRIX as TALIB_TRIX

            arr = s.to_numpy().astype(np.float64)
            result = TALIB_TRIX(arr, timeperiod=_length)
            return pl.Series(f"TRIX{_props}", result)

        trix_expr = close_expr.map_batches(compute_trix, return_dtype=pl.Float64)
    else:
        # Use pl_ema composition: triple EMA
        ema1 = ema(close_expr, length=length)
        ema2 = ema(ema1, length=length)
        ema3 = ema(ema2, length=length)

        # TRIX = scalar * pct_change(ema3, drift)
        ema3_shifted = ema3.shift(drift)
        trix_expr = scalar * (ema3 - ema3_shifted) / ema3_shifted

    # Signal = SMA of TRIX
    trix_signal_expr = trix_expr.rolling_mean(window_size=signal, min_samples=signal)

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
