# -*- coding: utf-8 -*-
from pandas import Series

from polars_ti.ma import ma
from polars_ti.utils import v_drift, v_mamode, v_offset, v_pos_default, v_series


def efi(
    close: Series,
    volume: Series,
    length: int | None = None,
    mamode: str | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Elder's Force Index (EFI)

    Elder's Force Index measures the power behind a price movement using
    price and volume as well as potential reversals and price corrections.

    Sources:
        https://www.tradingview.com/wiki/Elder%27s_Force_Index_(EFI)
        https://www.motivewave.com/studies/elders_force_index.htm

    Args:
        close (pd.Series): Series of 'close's
        volume (pd.Series): Series of 'volume's
        length (int): The short period. Default: 13
        drift (int): The diff period. Default: 1
        mamode (str): See ``help(ti.ma)``. Default: 'ema'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 13)
    close = v_series(close, length)
    volume = v_series(volume, length)

    if close is None or volume is None:
        return

    mamode = v_mamode(mamode, "ema")
    drift = v_drift(drift)
    offset = v_offset(offset)

    # Calculate
    pv_diff = close.diff(drift) * volume
    efi = ma(mamode, pv_diff, length=length)

    # Offset
    if offset != 0:
        efi = efi.shift(offset)

    # Fill
    if "fillna" in kwargs:
        efi = efi.fillna(kwargs["fillna"])

    # Name and Category
    efi.name = f"EFI_{length}"
    efi.category = "volume"

    return efi
