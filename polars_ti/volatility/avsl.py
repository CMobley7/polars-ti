# -*- coding: utf-8 -*-
from numpy import errstate, inf, nan, where
from pandas import Series

from polars_ti.overlap import sma
from polars_ti.utils import v_offset, v_pos_default, v_scalar, v_series


def avsl(
    close: Series,
    low: Series,
    volume: Series,
    fast_period: int | None = None,
    slow_period: int | None = None,
    scalar: float | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Anti-Volume Stop Loss (AVSL)

    AVSL is a volatility indicator that creates a trailing stop-loss line
    by combining volume-price analysis with dynamic support levels. It uses
    the Volume Price Confirmation Indicator (VPCI) methodology.

    The indicator works by:
    1. Computing VPCI components: Volume-Price Correlation, Volume-Price Ratio,
       and Volume Multiplier
    2. Adjusting price based on volume-price relationships
    3. Creating dynamic stop-loss levels using Bollinger Band-style deviation

    Based on Buff Dormeier's "Investing with Volume Analysis" methodology.

    Sources:
        https://www.tradingview.com/script/lYWz5r9e-AVSL-Anti-Volume-Stop-Loss/
        Dormeier, Buff. "Investing with Volume Analysis" (page 254)

    Args:
        close (pd.Series): Series of 'close's
        low (pd.Series): Series of 'low's
        volume (pd.Series): Series of 'volume's
        fast_period (int): Short period for VWMA and SMA. Default: 12
        slow_period (int): Long period for smoothing. Default: 26
        scalar (float): Band multiplier. Default: 2.0
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: AVSL stop-loss line
    """
    # Validate
    fast_period = v_pos_default(fast_period, 12)
    slow_period = v_pos_default(slow_period, 26)
    _length = slow_period + 1

    close = v_series(close, _length)
    low = v_series(low, _length)
    volume = v_series(volume, _length)

    if close is None or low is None or volume is None:
        return

    scalar = v_scalar(scalar, 2.0)
    offset = v_offset(offset)

    # Calculate VPCI components
    # Volume-Weighted and Simple Moving Averages
    # Late import to avoid circular dependency
    from polars_ti.volume import vwma

    vwma_fast = vwma(close, volume, length=fast_period)
    vwma_slow = vwma(close, volume, length=slow_period)
    sma_fast = sma(close, length=fast_period)
    sma_slow = sma(close, length=slow_period)

    if vwma_fast is None or vwma_slow is None or sma_fast is None or sma_slow is None:
        return Series(
            index=close.index,
            dtype=float,
            name=f"AVSL_{fast_period}_{slow_period}",
        )

    # Volume Price Confirmation Indicator (VPCI) components
    vpc = vwma_slow - sma_slow  # Volume Price Confirmation
    vpr = vwma_fast / sma_fast.replace(0, nan)  # Volume Price Ratio
    vm = (
        volume.rolling(fast_period).mean() / volume.rolling(slow_period).mean()
    )  # Volume Multiplier
    vpci = vpc * vpr * vm

    # Deviation based on VPCI
    deviation = scalar * vpci * vm

    # Adjust VPC to avoid division issues (clamp small values away from zero)
    vpc_adjusted = vpc.copy()
    vpc_adjusted = Series(
        where((vpc > -1) & (vpc < 0), -1, vpc_adjusted.to_numpy()),
        index=vpc.index,
    )
    vpc_adjusted = Series(
        where((vpc >= 0) & (vpc < 1), 1, vpc_adjusted.to_numpy()),
        index=vpc.index,
    )

    # Price function approximation
    with errstate(divide="ignore", invalid="ignore"):
        adjusted_price = low / (vpc_adjusted * vpr)
    adjusted_price = adjusted_price.replace([inf, -inf], nan)

    # Smoothed adjusted price
    price_function = adjusted_price.rolling(slow_period).mean() / 100

    # Final AVSL calculation
    raw_avsl = low - price_function + deviation
    avsl = raw_avsl.rolling(slow_period).mean()

    # Offset
    if offset != 0:
        avsl = avsl.shift(offset)

    # Fill
    if "fillna" in kwargs:
        avsl = avsl.fillna(kwargs["fillna"])

    # Name and Category
    avsl.name = f"AVSL_{fast_period}_{slow_period}"
    avsl.category = "volatility"

    return avsl
