# -*- coding: utf-8 -*-
from numpy import arctan, isnan, nan, zeros_like
from numba import njit


@njit(cache=True)
def nb_mama(x, fastlimit, slowlimit, prenan):
    a, b, m = 0.0962, 0.5769, x.size
    p_w, smp_w, smp_w_c = 0.2, 0.33, 0.67

    wma4 = zeros_like(x, dtype=x.dtype)
    dt, smp = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    i1, i2 = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    ji, jq = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    q1, q2 = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    re, im, alpha = (
        zeros_like(x, dtype=x.dtype),
        zeros_like(x, dtype=x.dtype),
        zeros_like(x, dtype=x.dtype),
    )
    period, phase = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    mama, fama = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)

    # Ehler's starts from 6, TV-LB from 3, TALib from 32
    for i in range(3, m):
        adj_prev_period = 0.075 * period[i - 1] + 0.54

        # WMA(x,4) & Detrended WMA(x,4)
        wma4[i] = 0.4 * x[i] + 0.3 * x[i - 1] + 0.2 * x[i - 2] + 0.1 * x[i - 3]
        dt[i] = adj_prev_period * (a * wma4[i] + b * wma4[i - 2] - b * wma4[i - 4] - a * wma4[i - 6])

        # Quadrature(Detrender) and In Phase Component
        q1[i] = adj_prev_period * (a * dt[i] + b * dt[i - 2] - b * dt[i - 4] - a * dt[i - 6])
        i1[i] = dt[i - 3]

        # Phase Q1 and I1 by 90 degrees
        ji[i] = adj_prev_period * (a * i1[i] + b * i1[i - 2] - b * i1[i - 4] - a * i1[i - 6])
        jq[i] = adj_prev_period * (a * q1[i] + b * q1[i - 2] - b * q1[i - 4] - a * q1[i - 6])

        # Phasor Addition for 3 Bar Averaging
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]

        # Smooth I2 & Q2
        i2[i] = p_w * i2[i] + (1 - p_w) * i2[i - 1]
        q2[i] = p_w * q2[i] + (1 - p_w) * q2[i - 1]

        # Homodyne Discriminator
        re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im[i] = i2[i] * q2[i - 1] + q2[i] * i2[i - 1]

        # Smooth Re & Im
        re[i] = p_w * re[i] + (1 - p_w) * re[i - 1]
        im[i] = p_w * im[i] + (1 - p_w) * im[i - 1]

        if im[i] != 0.0 and re[i] != 0.0:
            period[i] = 360 / arctan(im[i] / re[i])
        else:
            period[i] = 0

        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]
        if period[i] < 6:
            period[i] = 6
        if period[i] > 50:
            period[i] = 50

        period[i] = p_w * period[i] + (1 - p_w) * period[i - 1]
        smp[i] = smp_w * period[i] + smp_w_c * smp[i - 1]

        if i1[i] != 0.0:
            phase[i] = arctan(q1[i] / i1[i])

        dphase = phase[i - 1] - phase[i]
        if dphase < 1:
            dphase = 1

        alpha[i] = fastlimit / dphase
        if alpha[i] > fastlimit:
            alpha[i] = fastlimit
        if alpha[i] < slowlimit:
            alpha[i] = slowlimit

        mama[i] = alpha[i] * x[i] + (1 - alpha[i]) * mama[i - 1]
        fama[i] = 0.5 * alpha[i] * mama[i] + (1 - 0.5 * alpha[i]) * fama[i - 1]

    mama[:prenan], fama[:prenan] = nan, nan
    return mama, fama


# =============================================================================
# Polars MAMA Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.maps import Imports
from polars_ti.utils import v_talib
from polars_ti.utils._validate import v_expr


def mama(
    df: pl.DataFrame,
    close: str = "close",
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    prenan: int = 3,
    talib: bool = True,
    offset: int = 0,
) -> pl.DataFrame:
    """Polars: Ehler's MESA Adaptive Moving Average (MAMA)

    Args:
        df: Polars DataFrame with price columns
        close: Column name for 'close' prices. Default: "close"
        fastlimit: Fast limit. Default: 0.5
        slowlimit: Slow limit. Default: 0.05
        prenan: Prenans to apply. Default: 3
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.DataFrame: DataFrame with MAMA and FAMA columns
    """
    np_close = df.get_column(close).to_numpy().astype(np.float64)
    _props = f"_{fastlimit}_{slowlimit}"

    if Imports["talib"] and talib:
        from talib import MAMA as talib_mama

        mama_arr, fama_arr = talib_mama(np_close, fastlimit, slowlimit)
    else:
        mama_arr, fama_arr = nb_mama(np_close, fastlimit, slowlimit, prenan)

    if offset != 0:
        mama_arr = np.roll(mama_arr, offset)
        fama_arr = np.roll(fama_arr, offset)
        if offset > 0:
            mama_arr[:offset] = np.nan
            fama_arr[:offset] = np.nan

    return pl.DataFrame(
        {
            f"MAMA{_props}": mama_arr,
            f"FAMA{_props}": fama_arr,
        }
    )
