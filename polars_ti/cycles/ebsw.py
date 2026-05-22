# -*- coding: utf-8 -*-
# =============================================================================
# Polars EBSW Implementation
# =============================================================================
import polars as pl
from numpy import cos, exp, mean, pi, roll, sin, sqrt, zeros

from polars_ti._typing import IntoExpr


def pl_ebsw(
    close: str = "close",
    length: int = 40,
    bars: int = 10,
    initial_version: bool = False,
    offset: int = 0,
) -> callable:
    """Polars: Even Better SineWave (EBSW)

    Measures market cycles using a low pass filter to remove noise.
    Output is bounded between -1 and 1.

    This function returns a compute function that should be called with
    the DataFrame due to the recursive nature of the algorithm.

    Sources:
        - https://www.prorealcode.com/prorealtime-indicators/even-better-sinewave/
        - J.F.Ehlers 'Cycle Analytics for Traders', 2014

    Args:
        close: Column name for 'close' prices. Default: "close"
        length: Max cycle/trend period. Default: 40
        bars: Period of low pass filtering. Default: 10
        initial_version: Use initial version of algorithm. Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        callable: Function to apply to DataFrame
    """
    _offset = offset  # Capture for closure

    def compute_ebsw(df: pl.DataFrame) -> pl.DataFrame:
        close_arr = df[close].to_numpy()
        m = len(close_arr)

        lastHP = lastClose = 0.0
        filtHist = zeros(3)
        result = [float("nan")] * (length - 1) + [0.0]

        angle = 2 * pi / length
        alpha1 = (1 - sin(angle)) / cos(angle)
        ang = sqrt(2) * pi / bars
        a1 = exp(-ang)
        c2 = 2 * a1 * cos(ang)
        c3 = -(a1**2)
        c1 = 1 - c2 - c3

        for i in range(length, m):
            hp = 0.5 * (1 + alpha1) * (close_arr[i] - lastClose) + alpha1 * lastHP

            # Rotate filters
            filtHist = roll(filtHist, -1)
            filtHist[-1] = 0.5 * c1 * (hp + lastHP) + c2 * filtHist[1] + c3 * filtHist[0]

            # Wave calculation
            wave = mean(filtHist)
            rms = sqrt(mean(filtHist**2))
            if rms != 0:
                wave = wave / rms
            else:
                wave = 0.0

            lastHP = hp
            lastClose = close_arr[i]
            result.append(wave)

        result_df = pl.DataFrame({f"EBSW_{length}_{bars}": result})

        # Apply offset if needed
        if _offset != 0:
            result_df = result_df.select([pl.all().shift(_offset)])

        return result_df

    return compute_ebsw


def pl_ebsw_apply(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    """Apply EBSW to a DataFrame.

    Args:
        df: Polars DataFrame with close column
        **kwargs: Parameters (close, length, bars)

    Returns:
        pl.DataFrame: Original DataFrame with EBSW column added
    """
    close = kwargs.get("close", "close")
    length = kwargs.get("length", 40)
    bars = kwargs.get("bars", 10)

    compute_fn = pl_ebsw(close, length, bars)
    ebsw_df = compute_fn(df)
    return df.hstack(ebsw_df)
