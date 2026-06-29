# -*- coding: utf-8 -*-
from numpy import empty_like, maximum, minimum
from numba import njit


@njit(cache=True)
def np_ha(np_open, np_high, np_low, np_close):
    ha_close = 0.25 * (np_open + np_high + np_low + np_close)
    ha_open = empty_like(ha_close, dtype=np_open.dtype)
    ha_open[0] = 0.5 * (np_open[0] + np_close[0])

    m = np_close.size
    for i in range(1, m):
        ha_open[i] = 0.5 * (ha_open[i - 1] + ha_close[i - 1])

    ha_high = maximum(maximum(ha_open, ha_close), np_high)
    ha_low = minimum(minimum(ha_open, ha_close), np_low)

    return ha_open, ha_high, ha_low, ha_close


# =============================================================================
# Polars HA (Heikin-Ashi) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr


def ha(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Heikin-Ashi Candles

    The Heikin-Ashi technique averages price data to create smoothed
    candlestick charts that filter out market noise.

    Formula:
        HA_close = (open + high + low + close) / 4
        HA_open[0] = (open[0] + close[0]) / 2
        HA_open[i] = (HA_open[i-1] + HA_close[i-1]) / 2
        HA_high = max(high, HA_open, HA_close)
        HA_low = min(low, HA_open, HA_close)

    Sources:
        https://www.investopedia.com/terms/h/heikinashi.asp

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with HA_open, HA_high, HA_low, HA_close
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    _offset = offset

    def compute_ha(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        np_open = df["_open"].to_numpy().astype(np.float64)
        np_high = df["_high"].to_numpy().astype(np.float64)
        np_low = df["_low"].to_numpy().astype(np.float64)
        np_close = df["_close"].to_numpy().astype(np.float64)

        ha_open, ha_high, ha_low, ha_close = np_ha(np_open, np_high, np_low, np_close)

        if _offset != 0:
            ha_open = np.roll(ha_open, _offset)
            ha_high = np.roll(ha_high, _offset)
            ha_low = np.roll(ha_low, _offset)
            ha_close = np.roll(ha_close, _offset)
            if _offset > 0:
                ha_open[:_offset] = np.nan
                ha_high[:_offset] = np.nan
                ha_low[:_offset] = np.nan
                ha_close[:_offset] = np.nan
            else:
                ha_open[_offset:] = np.nan
                ha_high[_offset:] = np.nan
                ha_low[_offset:] = np.nan
                ha_close[_offset:] = np.nan

        n = len(np_close)
        return pl.Series(
            [
                {
                    "HA_open": ha_open[i],
                    "HA_high": ha_high[i],
                    "HA_low": ha_low[i],
                    "HA_close": ha_close[i],
                }
                for i in range(n)
            ]
        )

    fields = [
        pl.Field("HA_open", pl.Float64),
        pl.Field("HA_high", pl.Float64),
        pl.Field("HA_low", pl.Float64),
        pl.Field("HA_close", pl.Float64),
    ]

    return (
        pl.struct(
            [
                open_expr.alias("_open"),
                high_expr.alias("_high"),
                low_expr.alias("_low"),
                close_expr.alias("_close"),
            ]
        )
        .map_batches(compute_ha, return_dtype=pl.Struct(fields))
        .alias("HA")
    )


def ha_apply(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    """Apply Heikin-Ashi transformation to a DataFrame.

    Args:
        df: Polars DataFrame with OHLC columns
        **kwargs: Column names (open_, high, low, close)

    Returns:
        pl.DataFrame: Original DataFrame with HA columns added
    """
    open_ = kwargs.get("open_", "open")
    high = kwargs.get("high", "high")
    low = kwargs.get("low", "low")
    close = kwargs.get("close", "close")

    ha_struct = df.select(ha(open_, high, low, close))
    ha_df = ha_struct.unnest("HA")
    return df.hstack(ha_df)
