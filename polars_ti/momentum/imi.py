# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.utils import v_offset, v_pos_default, v_series


def imi(
    open_: Series,
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Intraday Momentum Index (IMI)

    The Intraday Momentum Index (IMI), developed by Tushar Chande, combines
    aspects of candlestick analysis with the Relative Strength Index (RSI)
    to generate overbought or oversold signals.

    IMI measures the relationship between opening and closing prices to gauge
    momentum within intraday sessions.

    Sources:
        https://www.investopedia.com/terms/i/intraday-momentum-index-imi.asp

    Calculation:
        Default Inputs:
            length=14
        Gains = Close - Open (when Close > Open, else 0)
        Losses = Open - Close (when Close < Open, else 0)
        Sum_Gains = sum(Gains, length)
        Sum_Losses = sum(Losses, length)
        IMI = 100 * Sum_Gains / (Sum_Gains + Sum_Losses)

    Args:
        open_ (pd.Series): Series of 'open's
        close (pd.Series): Series of 'close's
        length (int): The period for calculating sums. Default: 14
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: IMI values ranging from 0 to 100.
    """
    # Validate
    length = v_pos_default(length, 14)
    open_ = v_series(open_, length)
    close = v_series(close, length)
    offset = v_offset(offset)

    if open_ is None or close is None:
        return

    # Calculate
    # Gains when close > open, else 0
    gains = (close > open_) * (close - open_)
    # Losses when close < open, else 0
    losses = (close < open_) * (open_ - close)

    sum_gains = gains.rolling(length, min_periods=length).sum()
    sum_losses = losses.rolling(length, min_periods=length).sum()

    imi = 100 * sum_gains / (sum_gains + sum_losses)

    # Offset
    if offset != 0:
        imi = imi.shift(offset)

    # Fill
    if "fillna" in kwargs:
        imi = imi.fillna(kwargs["fillna"])

    # Name and Category
    imi.name = f"IMI_{length}"
    imi.category = "momentum"

    return imi
