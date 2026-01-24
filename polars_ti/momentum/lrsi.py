# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from pandas import Series

from polars_ti.utils import v_offset, v_pos_default, v_series


@njit(cache=True)
def nb_lrsi_filter(close, gamma):
    n = len(close)
    l0 = np.zeros(n)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    l3 = np.zeros(n)

    # Initialize with first value
    l0[0] = l1[0] = l2[0] = l3[0] = close[0]

    for i in range(1, n):
        l0[i] = (1 - gamma) * close[i] + gamma * l0[i - 1]
        l1[i] = -gamma * l0[i] + l0[i - 1] + gamma * l1[i - 1]
        l2[i] = -gamma * l1[i] + l1[i - 1] + gamma * l2[i - 1]
        l3[i] = -gamma * l2[i] + l2[i - 1] + gamma * l3[i - 1]

    return l0, l1, l2, l3


def lrsi(
    close: Series,
    length: int | None = None,
    gamma: float | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Laguerre RSI (LRSI)

    The Laguerre RSI is a modified RSI indicator that uses Laguerre polynomials
    to reduce lag and provide earlier signals. It adapts to price changes more
    quickly than the standard RSI while maintaining smooth oscillations.

    Sources:
        https://www.tradingview.com/script/3p0QrN5C-Laguerre-RSI/
        https://www.mesasoftware.com/papers/LaguerreFilters.pdf

    Calculation:
        Default Inputs:
            length=14, gamma=0.5

        Apply Laguerre filter with gamma coefficient:
        L0 = (1 - gamma) * Close + gamma * L0[1]
        L1 = -gamma * L0 + L0[1] + gamma * L1[1]
        L2 = -gamma * L1 + L1[1] + gamma * L2[1]
        L3 = -gamma * L2 + L2[1] + gamma * L3[1]

        Calculate ups and downs:
        CU = sum of (L0-L1, L1-L2, L2-L3) when positive
        CD = sum of (L0-L1, L1-L2, L2-L3) when negative (absolute)

        LRSI = 100 * CU / (CU + CD)

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 14
        gamma (float): Laguerre filter coefficient (0 to 1). Default: 0.5
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 14)
    gamma = float(gamma) if gamma and 0 < gamma < 1 else 0.5
    close = v_series(close, length)

    if close is None:
        return

    offset = v_offset(offset)

    # Calculate using numpy arrays for faster iteration
    close_arr = close.to_numpy(dtype=np.float64)
    n = len(close)

    # Apply Laguerre filter using Numba
    l0, l1, l2, l3 = nb_lrsi_filter(close_arr, gamma)

    # Calculate Laguerre RSI components (vectorized)
    cu = np.zeros(n)
    cd = np.zeros(n)

    cu += np.maximum(l0 - l1, 0)
    cd += np.maximum(l1 - l0, 0)
    cu += np.maximum(l1 - l2, 0)
    cd += np.maximum(l2 - l1, 0)
    cu += np.maximum(l2 - l3, 0)
    cd += np.maximum(l3 - l2, 0)

    # Calculate LRSI with division by zero protection
    denominator = cu + cd
    denominator = np.where(denominator == 0, 1, denominator)
    lrsi = Series(100 * cu / denominator, index=close.index)

    # Offset
    if offset != 0:
        lrsi = lrsi.shift(offset)

    # Fill
    if "fillna" in kwargs:
        lrsi = lrsi.fillna(kwargs["fillna"])

    # Name and Category
    lrsi.name = f"LRSI_{length}"
    lrsi.category = "momentum"

    return lrsi
