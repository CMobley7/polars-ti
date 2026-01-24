# -*- coding: utf-8 -*-
import numpy as np
from pandas import Series

from polars_ti.ma import ma
from polars_ti.utils import non_zero_range, v_mamode, v_offset, v_pos_default, v_series


def vfi(
    close: Series,
    volume: Series,
    length: int | None = None,
    coef: float | None = None,
    vcoef: float | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> Series:
    """Volume Flow Indicator (VFI)

    The Volume Flow Indicator (VFI) is a volume-based indicator that helps
    identify the strength of bulls vs bears in the market. It combines price
    movement with volume to show the flow of money into or out of a security.

    Sources:
        https://www.tradingview.com/script/MhlDpfdS-Volume-Flow-Indicator-LazyBear/
        https://www.investopedia.com/terms/v/volume-analysis.asp

    Calculation:
        Default Inputs:
            length=130, coef=0.2, vcoef=2.5, mamode='ema'

        typical = close
        inter = typical - typical.shift(1)  # Price change
        cutoff = coef * close  # Volatility threshold
        mf = inter if abs(inter) > cutoff else 0  # Filter minimal changes

        vave = SMA(volume, length).shift(1)
        vmax = vave * vcoef
        vc = min(volume, vmax)  # Clipped volume

        vcp = vc * mf  # Volume-weighted money flow

        VFI = SUM(vcp, length) / SMA(vave, length)
        VFI = EMA(VFI, 3)  # Smooth the result

    Args:
        close (pd.Series): Series of 'close's
        volume (pd.Series): Series of 'volume's
        length (int): The period. Default: 130
        coef (float): Volatility threshold coefficient. Default: 0.2
        vcoef (float): Volume coefficient. Default: 2.5
        mamode (str): Moving average mode for smoothing. Default: 'ema'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 130)
    coef = float(coef) if coef else 0.2
    vcoef = float(vcoef) if vcoef else 2.5
    mamode = v_mamode(mamode, "ema")
    close = v_series(close, length)
    volume = v_series(volume, length)

    if close is None or volume is None:
        return

    offset = v_offset(offset)

    # Calculate
    # Typical price (using close directly)
    typical = close

    # Volume cutoff
    vave = volume.rolling(length).mean().shift(1)
    vmax = vave * vcoef
    vc = Series(np.minimum(volume.values, vmax.values), index=volume.index)

    # Calculate MF (Money Flow) with volatility threshold
    inter = typical.diff(1)
    cutoff = coef * close
    mf = inter.where(inter.abs() > cutoff, 0)

    # Volume times cutoff price
    vcp = vc * mf

    # Calculate VFI with division by zero protection
    vave_sum = vave.rolling(length).mean()
    vave_sum = non_zero_range(vave_sum, vave_sum * 0)  # Protect against zero
    vfi = vcp.rolling(length).sum() / vave_sum

    # Smooth VFI
    vfi = ma(mamode, vfi, length=3)

    # Offset
    if offset != 0:
        vfi = vfi.shift(offset)

    # Fill
    if "fillna" in kwargs:
        vfi = vfi.fillna(kwargs["fillna"])

    # Name and Category
    vfi.name = f"VFI_{length}"
    vfi.category = "volume"

    return vfi
