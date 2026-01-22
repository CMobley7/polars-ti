# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.overlap import linreg
from polars_ti.utils import non_zero_range, v_offset, v_pos_default, v_series


def po(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Projection Oscillator (PO)

    The Projection Oscillator measures the percentage deviation of price from
    its linear regression trend line. It helps identify overbought and
    oversold conditions relative to the trend.

    Sources:
        https://www.tradingview.com/script/CDdh2vTz-Projection-Oscillator/
        Technical Analysis of Stock Trends by Edwards & Magee

    Calculation:
        Default Inputs:
            length=14

        LR = Linear Regression(close, length)
        PO = 100 * (close - LR) / LR

    Args:
        close (pd.Series): Series of 'close's
        length (int): The period. Default: 14
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 14)
    close = v_series(close, length)

    if close is None:
        return

    offset = v_offset(offset)

    # Calculate
    lr = linreg(close, length=length)

    # Projection oscillator as percentage with division protection
    lr_safe = non_zero_range(lr, lr * 0)
    po = 100 * (close - lr) / lr_safe

    # Offset
    if offset != 0:
        po = po.shift(offset)

    # Fill
    if "fillna" in kwargs:
        po = po.fillna(kwargs["fillna"])

    # Name and Category
    po.name = f"PO_{length}"
    po.category = "momentum"

    return po
