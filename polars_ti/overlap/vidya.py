# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from pandas import Series

from polars_ti.maps import Imports
from polars_ti.utils import v_drift, v_offset, v_pos_default, v_series, v_talib


@njit(cache=True)
def nb_vidya(close, abs_cmo, alpha, length):
    """Numba-optimized VIDYA calculation.

    State-dependent loop: each VIDYA value depends on previous VIDYA value.
    Values before 'length' are left as 0 and converted to NaN later.
    """
    m = close.size
    vidya = np.zeros(m)

    for i in range(length, m):
        vidya[i] = alpha * abs_cmo[i] * close[i] + vidya[i - 1] * (
            1 - alpha * abs_cmo[i]
        )
    return vidya


def vidya(
    close: Series,
    length: int | None = None,
    drift: int | None = None,
    offset: int | None = None,
    talib: bool | None = None,
    **kwargs: dict,
) -> Series:
    """Variable Index Dynamic Average (VIDYA)

    Variable Index Dynamic Average (VIDYA) was developed by Tushar Chande.
    It is similar to an EMA but it has a dynamically adjusted lookback
    period dependent on relative price volatility as measured by CMO. When
    volatility is high, VIDYA reacts faster to price changes.
    It is often used as moving average or trend identifier.

    Sources:
        https://www.tradingview.com/script/hdrf0fXV-Variable-Index-Dynamic-Average-VIDYA/
        https://www.perfecttrendsystem.com/blog_mt4_2/en/vidya-indicator-for-mt4

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 14
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 14)
    close = v_series(close, length + 1)

    if close is None:
        return

    mode_tal = v_talib(talib)
    drift = v_drift(drift)
    offset = v_offset(offset)

    # Calculate
    alpha = 2 / (length + 1)

    if Imports["talib"] and mode_tal:
        try:
            from talib import CMO

            cmo_ = CMO(close, length) / 100
        except ImportError:
            # Lazy import to avoid circular dependency
            from polars_ti.momentum.cmo import cmo

            cmo_ = cmo(close, length=length, drift=drift, talib=mode_tal)
    else:
        # Lazy import to avoid circular dependency
        from polars_ti.momentum.cmo import cmo

        cmo_ = cmo(close, length=length, drift=drift, talib=mode_tal)

    abs_cmo = cmo_.abs().astype(float)

    # Use Numba
    np_close = close.to_numpy(dtype=np.float64)
    np_abs_cmo = abs_cmo.to_numpy(dtype=np.float64)

    result = nb_vidya(np_close, np_abs_cmo, alpha, length)

    vidya = Series(result, index=close.index)
    vidya = vidya.replace({0: np.nan})

    # Offset
    if offset != 0:
        vidya = vidya.shift(offset)

    # Fill
    if "fillna" in kwargs:
        vidya = vidya.fillna(kwargs["fillna"])

    # Name and Category
    vidya.name = f"VIDYA_{length}"
    vidya.category = "overlap"

    return vidya
