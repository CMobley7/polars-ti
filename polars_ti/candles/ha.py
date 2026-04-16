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
from numpy import empty_like, maximum, minimum

from polars_ti._typing import IntoExpr


def pl_ha(
    open_: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    offset: int = 0,
) -> list[pl.Expr]:
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
        open_: Column name for 'open' prices. Default: "open"
        high: Column name for 'high' prices. Default: "high"
        low: Column name for 'low' prices. Default: "low"
        close: Column name for 'close' prices. Default: "close"
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: List of expressions for HA_open, HA_high, HA_low, HA_close
    """
    # HA_close is straightforward: average of OHLC
    ha_close_expr = (
        (pl.col(open_) + pl.col(high) + pl.col(low) + pl.col(close)) / 4
    ).alias("HA_close")

    # For HA_open, we need to use map_batches because it's recursive
    # HA_open[i] depends on HA_open[i-1] and HA_close[i-1]
    _offset = offset  # Capture for closure
    
    def compute_ha(df: pl.DataFrame) -> pl.DataFrame:
        np_open = df[open_].to_numpy()
        np_high = df[high].to_numpy()
        np_low = df[low].to_numpy()
        np_close = df[close].to_numpy()

        ha_close = 0.25 * (np_open + np_high + np_low + np_close)
        ha_open = empty_like(ha_close, dtype=np_open.dtype)
        ha_open[0] = 0.5 * (np_open[0] + np_close[0])

        m = np_close.size
        for i in range(1, m):
            ha_open[i] = 0.5 * (ha_open[i - 1] + ha_close[i - 1])

        ha_high = maximum(maximum(ha_open, ha_close), np_high)
        ha_low = minimum(minimum(ha_open, ha_close), np_low)

        result = pl.DataFrame({
            "HA_open": ha_open,
            "HA_high": ha_high,
            "HA_low": ha_low,
            "HA_close": ha_close,
        })
        
        # Apply offset if needed
        if _offset != 0:
            result = result.select([pl.all().shift(_offset)])
        
        return result

    return compute_ha


def pl_ha_apply(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
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

    compute_ha = pl_ha(open_, high, low, close)
    ha_df = compute_ha(df)
    return df.hstack(ha_df)

