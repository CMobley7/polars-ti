# -*- coding: utf-8 -*-
from numpy import arctan, copy, isnan, nan, rad2deg, zeros, zeros_like
from numba import njit


@njit(cache=True)
def nb_ht_trendline(x):
    a, b, m = 0.0962, 0.5769, x.size

    wma4, dt = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    q1, q2 = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    ji, jq = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    i1, i2 = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    re, im = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    period, smp = zeros_like(x, dtype=x.dtype), zeros_like(x, dtype=x.dtype)
    i_trend = zeros_like(x, dtype=x.dtype)

    result = zeros_like(x, dtype=x.dtype)
    result[:13] = x[:13]

    # Ehler's starts from 6, TALib from 63
    for i in range(6, m):
        adj_prev_period = 0.075 * period[i - 1] + 0.54

        wma4[i] = 0.4 * x[i] + 0.3 * x[i - 1] + 0.2 * x[i - 2] + 0.1 * x[i - 3]
        dt[i] = adj_prev_period * (a * wma4[i] + b * wma4[i - 2] - b * wma4[i - 4] - a * wma4[i - 6])

        q1[i] = adj_prev_period * (a * dt[i] + b * dt[i - 2] - b * dt[i - 4] - a * dt[i - 6])
        i1[i] = dt[i - 3]

        ji[i] = adj_prev_period * (a * i1[i] + b * i1[i - 2] - b * i1[i - 4] - a * i1[i - 6])
        jq[i] = adj_prev_period * (a * q1[i] + b * q1[i - 2] - b * q1[i - 4] - a * q1[i - 6])

        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]

        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]

        re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]

        re[i] = 0.2 * re[i] + 0.8 * re[i - 1]
        im[i] = 0.2 * im[i] + 0.8 * im[i - 1]

        if re[i] != 0 and im[i] != 0:
            period[i] = 360.0 / rad2deg(arctan(im[i] / re[i]))
        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]
        if period[i] < 6.0:
            period[i] = 6.0
        if period[i] > 50.0:
            period[i] = 50.0
        period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
        smp[i] = 0.33 * period[i] + 0.67 * smp[i - 1]

        dc_period = int(smp[i] + 0.5)
        dcp_avg = 0
        for k in range(dc_period):
            dcp_avg += x[i - k]

        if dc_period > 0:
            dcp_avg /= dc_period

        i_trend[i] = dcp_avg

        if i > 12:
            result[i] = 0.4 * i_trend[i] + 0.3 * i_trend[i - 1] + 0.2 * i_trend[i - 2] + 0.1 * i_trend[i - 3]

    return result


# =============================================================================
# Polars HT_Trendline Implementation (reuses nb_ht_trendline kernel)
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_ht_trendline(
    close: IntoExpr,
    prenan: int = 63,
    offset: int = 0,
) -> PlExpr:
    """Polars: Hilbert Transform TrendLine (HT_TL)

    Smooths price using the Hilbert Transform as described in
    Ehler's "Rocket Science for Traders".

    Args:
        close: Column name or pl.Expr for input values
        prenan: Prenans to apply. Default: 63
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: HT_TL expression
    """
    close_expr = v_expr(close)

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        result = nb_ht_trendline(arr)
        if prenan > 0:
            result[:prenan] = np.nan
        return pl.Series(values=result, name=s.name)

    result = close_expr.map_batches(_compute, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias("HT_TL")
