# -*- coding: utf-8 -*-
from numpy import sqrt, zeros
from numba import njit


@njit(cache=True)
def nb_hwc(close, na, nb, nc, nd, scalar):
    m = close.size
    result = np.zeros(m)
    upper = np.zeros(m)
    lower = np.zeros(m)

    last_a = last_v = last_var = 0.0
    last_f = last_price = last_result = float(close[0])

    for i in range(m):
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * close[i]
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        A = (1.0 - nc) * last_a + nc * (V - last_v)

        # Current result
        curr_res = F + V + 0.5 * A
        result[i] = curr_res

        var = (1.0 - nd) * last_var + nd * (last_price - last_result) * (last_price - last_result)
        stddev = np.sqrt(last_var)

        upper[i] = curr_res + scalar * stddev
        lower[i] = curr_res - scalar * stddev

        # update values
        last_price = close[i]
        last_a = A
        last_f = F
        last_v = V
        last_var = var
        last_result = curr_res

    return result, upper, lower


# =============================================================================
# Polars HWC Implementation (Numba @njit via map_batches)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_hwc(
    close: IntoExpr,
    scalar: float = 1.0,
    na: float = 0.2,
    nb: float = 0.1,
    nc: float = 0.1,
    nd: float = 0.1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: HWC (Holt-Winter Channel)

    Uses Numba @njit kernel via map_batches.

    Channel indicator based on HWMA - three-parameter moving average
    calculated by the method of Holt-Winters.

    Sources:
        https://www.mql5.com/en/code/20857

    Args:
        close: Column name or pl.Expr for 'close'
        scalar: Width multiplier of the channel. Default: 1
        na: Smoothed series (from 0 to 1). Default: 0.2
        nb: Trend value (from 0 to 1). Default: 0.1
        nc: Seasonality value (from 0 to 1). Default: 0.1
        nd: Channel value (from 0 to 1). Default: 0.1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with HWM (Mid), HWU (Upper), HWL (Lower) columns
    """
    close_expr = v_expr(close)

    if close_expr is None:
        return None

    _na = na
    _nb = nb
    _nc = nc
    _nd = nd
    _scalar = scalar
    _offset = offset

    def compute_hwc(s: pl.Series) -> pl.Series:
        np_close = s.to_numpy().astype(np.float64)
        result, upper, lower = nb_hwc(np_close, _na, _nb, _nc, _nd, _scalar)

        if _offset != 0:
            result = np.roll(result, _offset)
            upper = np.roll(upper, _offset)
            lower = np.roll(lower, _offset)
            if _offset > 0:
                result[:_offset] = np.nan
                upper[:_offset] = np.nan
                lower[:_offset] = np.nan

        return pl.DataFrame({"hwm": result, "hwu": upper, "hwl": lower}).to_struct("hwc")

    _props = f"_{int(scalar)}"

    return close_expr.map_batches(
        compute_hwc,
        return_dtype=pl.Struct({"hwm": pl.Float64, "hwu": pl.Float64, "hwl": pl.Float64}),
    ).alias(f"HWC{_props}")
