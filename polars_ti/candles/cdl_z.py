# -*- coding: utf-8 -*-
from pandas import DataFrame, Series

from polars_ti.statistics import zscore
from polars_ti.utils import v_bool, v_offset, v_pos_default, v_series


def cdl_z(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    full: bool | None = None,
    ddof: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """Candle Type: Z

    Normalizes OHLC Candles with a rolling Z Score.

    Source: Kevin Johnson

    Args:
        open_ (pd.Series): Series of 'open's
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        length (int): The period. Default: 10
        full (bool): Apply to whole DataFrame. Default: False
        ddof (int): Degrees of Freedom. Default: 1

    Kwargs:
        naive (bool, optional): If True, prefills potential Doji less than
            the length if less than a percentage of it's high-low range.
            Default: False
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: CDL_DOJI column.
    """
    # Validate
    length = v_pos_default(length, 30)
    open_ = v_series(open_, length)
    high = v_series(high, length)
    low = v_series(low, length)
    close = v_series(close, length)

    if open_ is None or high is None or low is None or close is None:
        return

    full = v_bool(full, False) if isinstance(full, bool) else False
    ddof = int(ddof) if isinstance(ddof, int) and 0 <= ddof < length else 1
    offset = v_offset(offset)

    # Calculate
    if full:
        length = close.size

    z_open = zscore(open_, length=length, ddof=ddof)
    z_high = zscore(high, length=length, ddof=ddof)
    z_low = zscore(low, length=length, ddof=ddof)
    z_close = zscore(close, length=length, ddof=ddof)

    _full = "a" if full else ""
    _props = _full if full else f"_{length}_{ddof}"
    data = {
        f"open_Z{_props}": z_open,
        f"high_Z{_props}": z_high,
        f"low_Z{_props}": z_low,
        f"close_Z{_props}": z_close,
    }
    df = DataFrame(data, index=close.index)

    if full:
        df = df.fillna(method="backfill", axis=0)

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Fill
    if "fillna" in kwargs:
        df = df.fillna(kwargs["fillna"])

    # Name and Category
    df.name = f"CDL_Z{_props}"
    df.category = "candles"

    return df
