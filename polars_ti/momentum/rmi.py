# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.ma import ma
from polars_ti.utils import (
    v_drift,
    v_mamode,
    v_offset,
    v_pos_default,
    v_scalar,
    v_series,
)


def rmi(
    close: Series,
    length: int | None = None,
    momentum: int | None = None,
    scalar: int | float | None = None,
    mamode: str | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Relative Momentum Index (RMI)

    The Relative Momentum Index is a momentum oscillator similar to RSI,
    but instead of comparing the current close to the previous close
    (momentum = 1), it compares to a close N periods ago. This makes it
    more responsive to momentum changes over longer periods.

    RMI was developed by Roger Altman and introduced in the February 1993
    issue of Technical Analysis of Stocks & Commodities magazine.

    Sources:
        https://www.investopedia.com/terms/r/relative_momentum_index.asp
        https://www.tradingview.com/script/DVd8Kzw4-Relative-Momentum-Index/

    Args:
        close (pd.Series): Series of 'close's
        length (int): The smoothing period. Default: 14
        momentum (int): The momentum lookback period. Default: 5
        scalar (float): How much to magnify. Default: 100
        mamode (str): See ``help(ti.ma)``. Default: 'rma'
        drift (int): The difference period (not used, for consistency). Default: 1
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: RMI values bounded between 0 and 100.
    """
    # Validate
    length = v_pos_default(length, 14)
    momentum = v_pos_default(momentum, 5)
    _length = length + momentum + 1
    close = v_series(close, _length)

    if close is None:
        return

    scalar = v_scalar(scalar, 100)
    mamode = v_mamode(mamode, "rma")
    drift = v_drift(drift)
    offset = v_offset(offset)

    # Calculate
    # Get momentum-based price changes (close now vs close N periods ago)
    momentum_change = close.diff(momentum)

    # Separate gains and losses
    gain = momentum_change.clip(lower=0)
    loss = (-momentum_change).clip(lower=0)

    # Smooth gains and losses using moving average
    avg_gain = ma(mamode, gain, length=length, **kwargs)
    avg_loss = ma(mamode, loss, length=length, **kwargs)

    # Calculate RMI using RSI formula: 100 - (100 / (1 + RS))
    rs = avg_gain / avg_loss
    rmi = scalar - (scalar / (1 + rs))

    # Offset
    if offset != 0:
        rmi = rmi.shift(offset)

    # Fill
    if "fillna" in kwargs:
        rmi = rmi.fillna(kwargs["fillna"])

    # Name and Category
    rmi.name = f"RMI_{length}_{momentum}"
    rmi.category = "momentum"

    return rmi
