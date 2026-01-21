# -*- coding: utf-8 -*-
from statistics import pstdev

from pandas import Series

from polars_ti.ma import ma
from polars_ti.utils import v_mamode, v_offset, v_pos_default, v_series


def vhm(
    volume: Series,
    length: int | None = None,
    slength=None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Volume Heatmap (VHM)

    Volume Heatmap is a volume indicator. It is used to indicate market/trend
    strength in a given time.

        Signal Table
        ==========================
        - extremely_cold <= -0.5
        - cold <= 1.0
        - medium <= 2.5
        - hot <= 4.0
        - extremely_hot >= 4+ (4 or more)

    Sources:
        https://www.tradingview.com/script/unWex8N4-Heatmap-Volume-xdecow/

    Args:
        volume (pd.Series): Series of 'volume's
        length (int): Length for mean calculation. Default: 610
        length (int): Length for standard devation calculation. Default: 610
        mamode (str): MA used to calculate the mean. See ```help(ti.ma)```.
            Default: 'sma'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 610)
    slength = v_pos_default(slength, length)
    _length = max(length, slength)
    volume = v_series(volume, _length)

    if volume is None:
        return

    mamode = v_mamode(mamode, "sma")
    offset = v_offset(offset)

    # Calculate
    mu = ma(mamode, volume, length=length)
    vhm = (volume - mu) / pstdev(volume, slength)

    # Offset
    if offset != 0:
        vhm = vhm.shift(offset)

    # Fill
    if "fillna" in kwargs:
        vhm = vhm.fillna(kwargs["fillna"])

    # Name and Category
    _props = f"VHM_{length}"
    vhm.name = _props if length == slength else f"{_props}_{slength}"
    vhm.category = "volume"

    return vhm
