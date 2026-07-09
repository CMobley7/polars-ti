# -*- coding: utf-8 -*-
# =============================================================================
# Polars ADX Implementation (Numba kernel)
# =============================================================================
import numpy as np
from numba import njit
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_adx(high, low, close, length, lensig, adxr_length, scalar):
    """Numba kernel for ADX, ADXR, DMP, DMN."""
    n = len(high)
    adx_out = np.full(n, np.nan)
    adxr_out = np.full(n, np.nan)
    dmp_out = np.full(n, np.nan)
    dmn_out = np.full(n, np.nan)

    # Guard: seed loops below index into arrays of size n at [0, length);
    # when n < length that reads/writes out of bounds. Return all-NaN.
    if n < length:
        return adx_out, adxr_out, dmp_out, dmn_out

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Plus/Minus DM
    pos = np.zeros(n)
    neg = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        if up > dn and up > 0:
            pos[i] = up
        if dn > up and dn > 0:
            neg[i] = dn

    # RMA smoothing for ATR, +DM, -DM
    atr_rma = np.zeros(n)
    pos_rma = np.zeros(n)
    neg_rma = np.zeros(n)

    # SMA init
    atr_sum = 0.0
    pos_sum = 0.0
    neg_sum = 0.0
    for i in range(length):
        atr_sum += tr[i]
        pos_sum += pos[i]
        neg_sum += neg[i]
    atr_rma[length - 1] = atr_sum / length
    pos_rma[length - 1] = pos_sum / length
    neg_rma[length - 1] = neg_sum / length

    alpha = 1.0 / length
    for i in range(length, n):
        atr_rma[i] = alpha * tr[i] + (1 - alpha) * atr_rma[i - 1]
        pos_rma[i] = alpha * pos[i] + (1 - alpha) * pos_rma[i - 1]
        neg_rma[i] = alpha * neg[i] + (1 - alpha) * neg_rma[i - 1]

    # DMP/DMN outputs are the Wilder sum-smoothed directional movement, matching
    # TA-Lib PLUS_DM/MINUS_DM (== length * RMA of the raw DM). DX is derived from
    # the directional indicators (+DI/-DI = scalar * smoothed_DM / smoothed_TR),
    # which are scale-invariant to the sum-vs-average smoothing choice.
    dx = np.full(n, np.nan)
    for i in range(length - 1, n):
        dmp_out[i] = length * pos_rma[i]
        dmn_out[i] = length * neg_rma[i]
        if atr_rma[i] != 0:
            di_pos = scalar * pos_rma[i] / atr_rma[i]
            di_neg = scalar * neg_rma[i] / atr_rma[i]
            di_sum = di_pos + di_neg
            if di_sum != 0:
                dx[i] = scalar * abs(di_pos - di_neg) / di_sum

    # ADX = RMA of DX
    # Find first valid dx
    first_valid = -1
    for i in range(n):
        if not np.isnan(dx[i]):
            first_valid = i
            break

    if first_valid >= 0 and first_valid + lensig - 1 < n:
        # SMA init for ADX
        dx_sum = 0.0
        for i in range(first_valid, first_valid + lensig):
            dx_sum += dx[i] if not np.isnan(dx[i]) else 0.0
        adx_out[first_valid + lensig - 1] = dx_sum / lensig

        a2 = 1.0 / lensig
        for i in range(first_valid + lensig, n):
            if not np.isnan(dx[i]):
                adx_out[i] = a2 * dx[i] + (1 - a2) * adx_out[i - 1]

    # ADXR
    for i in range(adxr_length, n):
        if not np.isnan(adx_out[i]) and not np.isnan(adx_out[i - adxr_length]):
            adxr_out[i] = 0.5 * (adx_out[i] + adx_out[i - adxr_length])

    return adx_out, adxr_out, dmp_out, dmn_out


def adx(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    lensig: int | None = None,
    adxr_length: int = 2,
    scalar: float = 100.0,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Average Directional Index (ADX)

    Quantifies trend strength by measuring directional movement.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: ATR/DM period. Default: 14
        lensig: Signal (ADX smoothing) period. Defaults to ``length`` when None.
        adxr_length: ADXR lookback. Default: 2
        scalar: Magnification. Default: 100
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with ADX, ADXR, DMP, DMN columns
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    # TA-Lib's ADX/ADXR expose only a single ``timeperiod``; the fast path is
    # therefore valid solely when the ADX smoothing period equals ``length``.
    # When the caller omits lensig it defaults to length (matching pandas-ta),
    # keeping the fast path active for the common case.
    _lensig = lensig if lensig is not None else length

    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    def _compute(s: pl.Series) -> pl.Series:
        data = s.struct.unnest()
        h = data["_h"].to_numpy().astype(np.float64)
        l_ = data["_l"].to_numpy().astype(np.float64)
        c = data["_c"].to_numpy().astype(np.float64)

        if Imports["talib"] and v_talib(talib) and length > 1 and _lensig == length:
            from talib import ADX, MINUS_DM, PLUS_DM

            adx_arr = ADX(h, l_, c, timeperiod=length)
            dmp_arr = PLUS_DM(h, l_, timeperiod=length)
            dmn_arr = MINUS_DM(h, l_, timeperiod=length)
            adxr_arr = 0.5 * (adx_arr + np.roll(adx_arr, adxr_length))
            adxr_arr[:adxr_length] = np.nan
        else:
            adx_arr, adxr_arr, dmp_arr, dmn_arr = _nb_adx(h, l_, c, length, _lensig, adxr_length, scalar)

        if offset != 0:
            for arr in [adx_arr, adxr_arr, dmp_arr, dmn_arr]:
                arr[:] = np.roll(arr, offset)
                if offset > 0:
                    arr[:offset] = np.nan
                else:
                    arr[offset:] = np.nan

        n = len(h)
        return pl.Series(
            values=[
                {
                    f"ADX_{_lensig}": adx_arr[i],
                    f"ADXR_{_lensig}_{adxr_length}": adxr_arr[i],
                    f"DMP_{length}": dmp_arr[i],
                    f"DMN_{length}": dmn_arr[i],
                }
                for i in range(n)
            ]
        )

    fields = [
        pl.Field(f"ADX_{_lensig}", pl.Float64),
        pl.Field(f"ADXR_{_lensig}_{adxr_length}", pl.Float64),
        pl.Field(f"DMP_{length}", pl.Float64),
        pl.Field(f"DMN_{length}", pl.Float64),
    ]
    return (
        pl.struct(
            high_expr.alias("_h"),
            low_expr.alias("_l"),
            close_expr.alias("_c"),
        )
        .map_batches(_compute, return_dtype=pl.Struct(fields))
        .alias(f"ADX_{_lensig}")
    )
