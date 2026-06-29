# -*- coding: utf-8 -*-
# =============================================================================
# Polars QQE (Quantitative Qualitative Estimation) Implementation
# =============================================================================
import numpy as np
import polars as pl
from numba import jit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.momentum.rsi import rsi
from polars_ti.ma import ma


@jit(nopython=True, cache=True)
def nb_qqe_loop(
    rsi_ma: np.ndarray,
    upperband: np.ndarray,
    lowerband: np.ndarray,
) -> tuple:
    """Numba-optimized QQE stateful loop.

    Calculates long/short/trend/qqe/qqe_long/qqe_short arrays.
    """
    n = len(rsi_ma)
    long = np.zeros(n, dtype=np.float64)
    short = np.zeros(n, dtype=np.float64)
    trend = np.ones(n, dtype=np.float64)
    qqe = np.empty(n, dtype=np.float64)
    qqe_long = np.empty(n, dtype=np.float64)
    qqe_short = np.empty(n, dtype=np.float64)

    qqe[:] = np.nan
    qqe_long[:] = np.nan
    qqe_short[:] = np.nan

    # Initialize first valid value
    for i in range(n):
        if not np.isnan(rsi_ma[i]):
            qqe[i] = rsi_ma[i]
            break

    for i in range(1, n):
        if np.isnan(rsi_ma[i]) or np.isnan(rsi_ma[i - 1]):
            continue

        c_rsi = rsi_ma[i]
        p_rsi = rsi_ma[i - 1]
        c_long = long[i - 1]
        p_long = long[max(0, i - 2)]
        c_short = short[i - 1]
        p_short = short[max(0, i - 2)]

        # Long Line
        if p_rsi > c_long and c_rsi > c_long:
            long[i] = max(c_long, lowerband[i])
        else:
            long[i] = lowerband[i]

        # Short Line
        if p_rsi < c_short and c_rsi < c_short:
            short[i] = min(c_short, upperband[i])
        else:
            short[i] = upperband[i]

        # Trend & QQE Calculation
        if (c_rsi > c_short and p_rsi < p_short) or (c_rsi <= c_short and p_rsi >= p_short):
            trend[i] = 1
            qqe[i] = qqe_long[i] = long[i]
        elif (c_rsi > c_long and p_rsi < p_long) or (c_rsi <= c_long and p_rsi >= p_long):
            trend[i] = -1
            qqe[i] = qqe_short[i] = short[i]
        else:
            trend[i] = trend[i - 1]
            if trend[i] == 1:
                qqe[i] = qqe_long[i] = long[i]
            else:
                qqe[i] = qqe_short[i] = short[i]

    return qqe, qqe_long, qqe_short


def qqe(
    close: IntoExpr,
    length: int = 14,
    smooth: int = 5,
    factor: float = 4.236,
    mamode: str = "ema",
    offset: int = 0,
) -> PlExpr:
    """Polars: Quantitative Qualitative Estimation (QQE)

    QQE is a momentum indicator that combines RSI with volatility-based
    trailing stop lines (similar to ATR bands) to identify trend direction
    and potential reversals.

    Sources:
        https://www.tradingview.com/script/IYfA9R2k-QQE-MOD/

    Calculation:
        1. RSI = rsi(close, length)
        2. RSI_MA = ma(mamode, RSI, smooth)
        3. RSI_TR = abs(RSI_MA.diff())
        4. DAR = factor * ema(ema(RSI_TR, 2*length-1), 2*length-1)
        5. Upper/Lower bands = RSI_MA ± DAR
        6. Stateful loop for long/short/trend/qqe lines

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: RSI period. Default: 14
        smooth: MA smoothing period for RSI. Default: 5
        factor: ATR multiplier for bands. Default: 4.236
        mamode: MA type for RSI smoothing. Default: 'ema'
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with QQE, QQEl, QQEs columns
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    wilders_length = 2 * length - 1
    _mode = mamode.lower()[0] if mamode != "ema" else ""
    _props = f"{_mode}_{length}_{smooth}_{factor}"
    _rsima_name = f"QQE{_props}_RSI{_mode.upper()}MA"

    # Capture loop variables for closure
    _length = length
    _smooth = smooth
    _factor = factor
    _mamode = mamode
    _wilders_length = wilders_length
    _props_str = _props

    def compute_qqe(s: pl.Series) -> pl.DataFrame:
        """Compute QQE using pure Polars/Numba — no pandas dependency."""
        arr = s.to_numpy().astype(np.float64)
        tmp = pl.DataFrame({"_close": arr})

        # Step 1: RSI via pl_rsi evaluated eagerly
        rsi_col = tmp.select(rsi("_close", length=_length)).to_series().to_numpy()

        # Step 2: Smooth RSI with pl_ma
        tmp2 = pl.DataFrame({"_rsi": rsi_col})
        rsi_ma_expr = ma(_mamode, "_rsi", length=_smooth)
        rsi_ma_col = tmp2.select(rsi_ma_expr).to_series().to_numpy()

        # Step 3: RSI MA True Range
        rsi_ma_tr = np.abs(np.diff(rsi_ma_col, prepend=np.nan))

        # Step 4: Double EMA of RSI TR
        tmp3 = pl.DataFrame({"_tr": rsi_ma_tr})
        s1 = tmp3.select(ma("ema", "_tr", length=_wilders_length)).to_series().to_numpy()
        tmp4 = pl.DataFrame({"_s1": s1})
        s2 = tmp4.select(ma("ema", "_s1", length=_wilders_length)).to_series().to_numpy()
        dar = _factor * s2

        # Step 5: Upper/Lower bands
        upperband = rsi_ma_col + dar
        lowerband = rsi_ma_col - dar

        # Step 6: Run Numba loop
        qqe_vals, qqe_long_vals, qqe_short_vals = nb_qqe_loop(rsi_ma_col, upperband, lowerband)

        return pl.DataFrame(
            {
                f"QQE{_props_str}": qqe_vals,
                _rsima_name: rsi_ma_col,
                f"QQEl{_props_str}": qqe_long_vals,
                f"QQEs{_props_str}": qqe_short_vals,
            }
        )

    result = close_expr.map_batches(
        lambda s: compute_qqe(s).to_struct("QQE"),
        return_dtype=pl.Struct(
            [
                pl.Field(f"QQE{_props}", pl.Float64),
                pl.Field(_rsima_name, pl.Float64),
                pl.Field(f"QQEl{_props}", pl.Float64),
                pl.Field(f"QQEs{_props}", pl.Float64),
            ]
        ),
    ).alias(f"QQE{_props}")

    if offset != 0:
        result = result.shift(offset)

    return result
