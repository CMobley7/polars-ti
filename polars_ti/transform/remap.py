# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.utils import v_float, v_offset, v_series


def remap(
    close: Series,
    from_min: int | float | None = None,
    from_max: int | float | None = None,
    to_min: int | float | None = None,
    to_max: int | float | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """
    Indicator: ReMap (REMAP)

    Basically a static normalizer, which maps the input min and max to a given
    output range. Many range bound oscillators move between 0 and 100, but
    there are also other variants. Refer to the example below or add more the
    list.

    Examples:
        RSI -> IFISHER: from_min=0, from_max=100, to_min=-1, to_max=1.0

    Sources:
        rengel8 for Polars TI

    Args:
        close (pd.Series): Series of 'close's
        from_min (float): Input minimum. Default: 0.0
        from_max (float): Input maximum. Default: 100.0
        to_min (float): Output minimum. Default: 0.0
        to_max (float): Output maximum. Default: 100.0
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    close = v_series(close)
    from_min = v_float(from_min, 0.0, 0.0)
    from_max = v_float(from_max, 100.0, 0.0)
    to_min = v_float(to_min, -1.0, 0.0)
    to_max = v_float(to_max, 1.0, 0.0)
    offset = v_offset(offset)

    # Calculate
    frange, trange = from_max - from_min, to_max - to_min
    if frange <= 0 or trange <= 0:
        return
    result = to_min + (trange / frange) * (close.to_numpy() - from_min)
    result = Series(result, index=close.index)

    # Offset
    if offset != 0:
        result = result.shift(offset)

    # Fill
    if "fillna" in kwargs:
        result = result.fillna(kwargs["fillna"])

    # Name and Category
    result.name = f"REMAP_{from_min}_{from_max}_{to_min}_{to_max}"
    # result.name = f"{close.name}_{from_min}_{from_max}_{to_min}_{to_max}" # OR
    result.category = "transform"

    return result
