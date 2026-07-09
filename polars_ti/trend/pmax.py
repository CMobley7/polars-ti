# -*- coding: utf-8 -*-
# =============================================================================
# Polars PMAX Implementation
# =============================================================================
import numpy as np
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_pmax(close, ub, lb):
    m = close.size
    dir_ = np.ones(m)  # 1 for Uptrend, -1 for Downtrend
    trend = np.full(m, np.nan)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)

    # State-dependent iteration
    for i in range(1, m):
        # Trend detection: price crosses bands
        if close[i] > lb[i - 1]:
            dir_[i] = 1  # Uptrend
        elif close[i] < ub[i - 1]:
            dir_[i] = -1  # Downtrend
        else:
            dir_[i] = dir_[i - 1]  # Maintain previous trend

            # Adjust bands to not move against trend
            if dir_[i] > 0 and ub[i] < ub[i - 1]:
                ub[i] = ub[i - 1]
            if dir_[i] < 0 and lb[i] > lb[i - 1]:
                lb[i] = lb[i - 1]

        # Set PMAX value based on trend direction
        if dir_[i] > 0:
            trend[i] = long[i] = ub[i]
        else:
            trend[i] = short[i] = lb[i]

    return trend, dir_, long, short


def pmax(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr | None = None,
    length: int = 10,
    multiplier: float = 3.0,
    mamode: str = "ema",
    talib: bool = False,
    offset: int = 0,
) -> PlExpr:
    """Polars: PMAX (Price Max)

    Combines ATR-based volatility bands with a moving average for
    adaptive trailing stops.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        volume: Unused (accepted for accessor compatibility). Default: None
        length: ATR/MA period. Default: 10
        multiplier: ATR multiplier. Default: 3.0
        mamode: MA type ('ema' or 'sma'). Default: 'ema'
        talib: If True and TA-Lib installed, use TA-Lib for ATR. Default: False
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with PMAX, PMAXd, PMAXl, PMAXs columns
    """
    from polars_ti.ma import ma
    from polars_ti.utils import v_pos_int
    from polars_ti.volatility.atr import atr

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    length = v_pos_int(length, "length")

    # Composition: shared ATR + MA (native seeding matches pandas-ta in WS3)
    atr_expr = atr(high_expr, low_expr, close_expr, length=length, talib=talib)
    ma_expr = ma(mamode, close_expr, length=length, talib=talib)

    matr = multiplier * atr_expr
    ub_expr = ma_expr - matr  # Upper band (support in uptrend)
    lb_expr = ma_expr + matr  # Lower band (resistance in downtrend)

    def _compute(s: pl.Series) -> pl.Series:
        data = s.struct.unnest()
        c = data["_c"].to_numpy().astype(np.float64)
        ub = data["_ub"].to_numpy().astype(np.float64)
        lb = data["_lb"].to_numpy().astype(np.float64)

        trend, dir_, long_arr, short_arr = nb_pmax(c, ub, lb)
        dir_[:length] = np.nan

        if offset != 0:
            for a in [trend, dir_, long_arr, short_arr]:
                a[:] = np.roll(a, offset)
                if offset > 0:
                    a[:offset] = np.nan
                else:
                    a[offset:] = np.nan

        _props = f"_{length}_{multiplier}"
        n = len(c)
        return pl.Series(
            values=[
                {
                    f"PMAX{_props}": trend[i],
                    f"PMAXd{_props}": dir_[i],
                    f"PMAXl{_props}": long_arr[i],
                    f"PMAXs{_props}": short_arr[i],
                }
                for i in range(n)
            ]
        )

    _pprops = f"_{length}_{multiplier}"
    fields = [
        pl.Field(f"PMAX{_pprops}", pl.Float64),
        pl.Field(f"PMAXd{_pprops}", pl.Float64),
        pl.Field(f"PMAXl{_pprops}", pl.Float64),
        pl.Field(f"PMAXs{_pprops}", pl.Float64),
    ]
    return (
        pl.struct(
            close_expr.alias("_c"),
            ub_expr.alias("_ub"),
            lb_expr.alias("_lb"),
        )
        .map_batches(_compute, return_dtype=pl.Struct(fields))
        .alias(f"PMAX_{length}_{multiplier}")
    )
