# -*- coding: utf-8 -*-
from pandas import DataFrame, Series

from polars_ti.momentum import trix
from polars_ti.utils import v_drift, v_offset, v_pos_default, v_scalar, v_series


def trixh(
    close: Series,
    length: int | None = None,
    signal: int | None = None,
    scalar: float | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: dict,
) -> DataFrame:
    """TRIX Histogram (TRIXH)

    TRIX Histogram extends the TRIX indicator by adding a signal line and
    histogram. The histogram represents the difference between TRIX and its
    signal line, similar to MACD histogram, helping identify momentum changes
    and divergences.

    Sources:
        https://www.investopedia.com/terms/t/trix.asp
        https://school.stockcharts.com/doku.php?id=technical_indicators:trix

    Calculation:
        Default Inputs:
            length=18, signal=9, scalar=100

        TRIX = TRIX(close, length, scalar)
        Signal = EMA(TRIX, signal)
        Histogram = TRIX - Signal

    Args:
        close (pd.Series): Series of 'close's
        length (int): TRIX period. Default: 18
        signal (int): Signal line period. Default: 9
        scalar (float): Multiplier. Default: 100
        drift (int): The difference period. Default: 1
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.DataFrame: TRIX, Signal, and Histogram columns.
    """
    # Validate
    length = v_pos_default(length, 18)
    signal_length = v_pos_default(signal, 9)
    scalar = v_scalar(scalar, 100)
    close = v_series(close, length)

    if close is None:
        return

    drift = v_drift(drift)
    offset = v_offset(offset)

    # Calculate TRIX (returns DataFrame with TRIX and signal)
    trix_df = trix(
        close, length=length, signal=signal_length, scalar=scalar, drift=drift
    )

    if trix_df is None:
        return

    # Extract TRIX line and signal
    trix_col = f"TRIX_{length}_{signal_length}"
    signal_col = f"TRIXs_{length}_{signal_length}"

    trix_line = trix_df[trix_col]
    trix_signal = trix_df[signal_col]

    # Calculate histogram
    histogram = trix_line - trix_signal

    # Offset
    if offset != 0:
        trix_line = trix_line.shift(offset)
        trix_signal = trix_signal.shift(offset)
        histogram = histogram.shift(offset)

    # Fill
    if "fillna" in kwargs:
        trix_line = trix_line.fillna(kwargs["fillna"])
        trix_signal = trix_signal.fillna(kwargs["fillna"])
        histogram = histogram.fillna(kwargs["fillna"])

    # Name and Category
    trix_line.name = f"TRIX_{length}_{signal_length}"
    trix_signal.name = f"TRIXs_{length}_{signal_length}"
    histogram.name = f"TRIXh_{length}_{signal_length}"
    trix_line.category = trix_signal.category = histogram.category = "momentum"

    data = {
        trix_line.name: trix_line,
        trix_signal.name: trix_signal,
        histogram.name: histogram,
    }
    df = DataFrame(data, index=close.index)
    df.name = f"TRIXH_{length}_{signal_length}"
    df.category = "momentum"

    return df
