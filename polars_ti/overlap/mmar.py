# -*- coding: utf-8 -*-
from pandas import DataFrame, Series

from polars_ti.overlap.ema import ema
from polars_ti.utils import v_offset, v_pos_default, v_series


def mmar(
    close: Series,
    length: int | None = None,
    step: int | None = None,
    num_ribbons: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """Madrid Moving Average Ribbon (MMAR)

    The Madrid Moving Average Ribbon is a visual trend indicator that consists
    of multiple EMAs with incrementally increasing periods. It helps identify
    trend strength and direction through the spacing and alignment of the
    moving averages.

    Sources:
        https://www.tradingview.com/script/a87v7d4L-Madrid-Moving-Average-Ribbon/
        https://www.forexstrategiesresources.com/trend-following-forex-strategies/

    Calculation:
        Default Inputs:
            length=10, step=5, num_ribbons=6

        For i in range(num_ribbons):
            period = length + (i * step)
            MMAR[i] = EMA(close, period)

        Returns DataFrame with columns:
        MMAR_10, MMAR_15, MMAR_20, MMAR_25, MMAR_30, MMAR_35

    Args:
        close (pd.Series): Series of 'close's
        length (int): Initial EMA period. Default: 10
        step (int): Period increment between ribbons. Default: 5
        num_ribbons (int): Number of EMA ribbons. Default: 6
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: New features generated.
    """
    # Validate
    length = v_pos_default(length, 10)
    step = v_pos_default(step, 5)
    num_ribbons = v_pos_default(num_ribbons, 6)
    _max_length = length + (num_ribbons - 1) * step
    close = v_series(close, _max_length)

    if close is None:
        return

    offset = v_offset(offset)

    # Calculate - Create ribbon of EMAs with incremental periods
    ribbons = {}
    for i in range(num_ribbons):
        period = length + (i * step)
        ema_value = ema(close, length=period)
        ribbons[f"MMAR_{period}"] = ema_value

    df = DataFrame(ribbons, index=close.index)

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Fill
    if "fillna" in kwargs:
        df = df.fillna(kwargs["fillna"])

    # Name and Category
    df.name = f"MMAR_{length}_{step}_{num_ribbons}"
    df.category = "overlap"

    return df
