# -*- coding: utf-8 -*-
# =============================================================================
# Polars Implementation
# =============================================================================
from sys import float_info as sflt

import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.ma import pl_ma


def pl_smc(
    open_: IntoExpr = "open",
    high: IntoExpr = "high",
    low: IntoExpr = "low",
    close: IntoExpr = "close",
    abr_length: int = 14,
    close_length: int = 50,
    vol_length: int = 20,
    percent: int = 5,
    vol_ratio: float = 1.5,
    asint: bool = True,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Smart Money Concept (SMC)

    The Smart Money concept combines several techniques to identify significant
    price movements that might indicate 'smart money' actions. It uses candlestick
    patterns, moving averages, and imbalance calculations.

    Sources:
        https://www.tradingview.com/script/CnB3fSph-Smart-Money-Concepts-LuxAlgo/

    Args:
        open_ (IntoExpr): Column name or expression for open. Default: "open"
        high (IntoExpr): Column name or expression for high. Default: "high"
        low (IntoExpr): Column name or expression for low. Default: "low"
        close (IntoExpr): Column name or expression for close. Default: "close"
        abr_length (int): Average Bar Range (abr) window length. Default: 14
        close_length (int): The moving average length for 'close'. Default: 50
        vol_length (int): The length for calculating volatility. Default: 20
        percent (int): Percent of wick that exceeds the body. Default: 5
        vol_ratio (float): Volatility ratio to determine high volatility condition.
            Default: 1.5
        asint (bool): Keep results numerical instead of boolean. Default: True
        mamode (str): See pl_ma(). Default: 'sma'
        talib (bool): If TA Lib is installed and talib is True, uses TA Lib.
            Default: True
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pl.Expr: Struct expression with columns:
            SMChv - High Volatility flag
            SMCbf - Bottom Imbalance Flag
            SMCbi - Bottom Imbalance Value
            SMCbp - Bottom Imbalance Percent
            SMCtf - Top Imbalance Flag
            SMCti - Top Imbalance Value
            SMCtp - Top Imbalance Percent
    """
    # Handle swapping if close_length < abr_length
    if close_length < abr_length:
        abr_length, close_length = close_length, abr_length

    _props = f"_{abr_length}_{close_length}_{vol_length}_{percent}"
    eps = sflt.epsilon

    # Convert inputs to expressions
    open_col = pl.col(open_) if isinstance(open_, str) else open_
    high_col = pl.col(high) if isinstance(high, str) else high
    low_col = pl.col(low) if isinstance(low, str) else low
    close_col = pl.col(close) if isinstance(close, str) else close

    # Body calculations
    body_high = pl.max_horizontal(open_col, close_col)
    body_low = pl.min_horizontal(open_col, close_col)
    body = body_high - body_low + eps

    # ABR (Average Bar Range) - rolling max of high minus rolling min of low
    abr = (
        high_col.rolling_max(window_size=abr_length)
        - low_col.rolling_min(window_size=abr_length)
    )

    # Imbalance calculations (FVG - Fair Value Gap)
    # top_imbalance = low[i-2] - high[i] (bearish imbalance)
    # btm_imbalance = low[i] - high[i-2] (bullish imbalance)
    top_imbalance = low_col.shift(2) - high_col
    btm_imbalance = low_col - high_col.shift(2)

    # Imbalance percentages
    top_imbalance_pct = 100.0 * top_imbalance / abr
    btm_imbalance_pct = 100.0 * btm_imbalance / abr

    # High-Low Delta for volatility calculation
    hld = high_col - low_col + eps

    # High volatility: HLD > vol_ratio * MA(HLD)
    hld_ma = pl_ma(name=mamode, source=hld, length=vol_length, talib=talib)
    high_volatility = hld > (vol_ratio * hld_ma)

    # Imbalance flags
    btm_imbalance_flag = (btm_imbalance > 0) & (btm_imbalance_pct > 1)
    top_imbalance_flag = (top_imbalance > 0) & (top_imbalance_pct > 1)

    # Convert to int if asint
    if asint:
        high_volatility = high_volatility.cast(pl.Int64)
        btm_imbalance_flag = btm_imbalance_flag.cast(pl.Int64)
        top_imbalance_flag = top_imbalance_flag.cast(pl.Int64)

    # Apply offset
    if offset != 0:
        high_volatility = high_volatility.shift(offset)
        btm_imbalance_flag = btm_imbalance_flag.shift(offset)
        btm_imbalance = btm_imbalance.shift(offset)
        btm_imbalance_pct = btm_imbalance_pct.shift(offset)
        top_imbalance_flag = top_imbalance_flag.shift(offset)
        top_imbalance = top_imbalance.shift(offset)
        top_imbalance_pct = top_imbalance_pct.shift(offset)

    # Apply ffill then bfill
    high_volatility = high_volatility.forward_fill().backward_fill()
    btm_imbalance_flag = btm_imbalance_flag.forward_fill().backward_fill()
    btm_imbalance = btm_imbalance.forward_fill().backward_fill()
    btm_imbalance_pct = btm_imbalance_pct.forward_fill().backward_fill()
    top_imbalance_flag = top_imbalance_flag.forward_fill().backward_fill()
    top_imbalance = top_imbalance.forward_fill().backward_fill()
    top_imbalance_pct = top_imbalance_pct.forward_fill().backward_fill()

    # Return as struct
    return pl.struct(
        high_volatility.alias(f"SMChv{_props}"),
        btm_imbalance_flag.alias(f"SMCbf{_props}"),
        btm_imbalance.alias(f"SMCbi{_props}"),
        btm_imbalance_pct.alias(f"SMCbp{_props}"),
        top_imbalance_flag.alias(f"SMCtf{_props}"),
        top_imbalance.alias(f"SMCti{_props}"),
        top_imbalance_pct.alias(f"SMCtp{_props}"),
    ).alias(f"SMC{_props}")
