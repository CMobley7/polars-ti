# -*- coding: utf-8 -*-
import re as re_
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from sys import float_info as sflt

from numba import njit
from numpy import argmax, argmin, finfo, float64
from pandas import DataFrame, Series

from polars_ti.maps import Imports
from polars_ti.utils._validate import v_bool, v_dataframe, v_pos_default, v_series


def camelCase2Title(x: str):
    """https://stackoverflow.com/questions/5020906/python-convert-camel-case-to-space-delimited-using-regex-and-taking-acronyms-in"""
    return re_.sub("([a-z])([A-Z])", r"\g<1> \g<2>", x).title()


def category_files(category: str) -> list:
    """Helper function to return all filenames in the category directory."""
    files = [
        x.stem
        for x in list(Path(f"polars_ti/{category}/").glob("*.py"))
        if x.stem != "__init__"
    ]
    return files


def non_zero_range(high: Series, low: Series) -> Series:
    """Returns the difference of two series and adds epsilon to any zero values.
    This occurs commonly in crypto data when 'high' = 'low'."""
    diff = high - low
    if diff.eq(0).any().any():
        diff += sflt.epsilon
    return diff


@njit(cache=True)
def nb_non_zero_range(x, y):
    diff = x - y
    if diff.any() == 0:
        diff += finfo(float64).eps
    return diff


def recent_maximum_index(x) -> int:
    return int(argmax(x[::-1]))


def recent_minimum_index(x) -> int:
    return int(argmin(x[::-1]))


def rma_pandas(series: Series, length: int):
    series = v_series(series)
    alpha = (1.0 / length) if length > 0 else 0.5
    return series.ewm(alpha=alpha, min_periods=length).mean()


def signed_series(series: Series, initial: int, lag: int | None = None) -> Series:
    """Returns a Signed Series with or without an initial value

    Default Example:
    series = Series([3, 2, 2, 1, 1, 5, 6, 6, 7, 5])
    and returns:
    sign = Series([nan, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0])
    """
    initial = None
    if initial is not None and not isinstance(lag, str):
        initial = initial
    series = v_series(series)
    lag = v_pos_default(lag, 1)
    sign = series.diff(lag)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = initial
    return sign


def simplify_columns(df, n: int = 3) -> list[str]:
    df.columns = df.columns.str.lower()
    return [c.split("_")[0][n - 1 : n] for c in df.columns]


def tal_ma(name: str) -> int:
    """Helper Function that returns the Enum value for TA Lib's MA Type"""
    if Imports["talib"] and isinstance(name, str) and len(name) > 1:
        from talib import MA_Type

        name = name.lower()
        if name == "sma":
            return MA_Type.SMA  # 0
        elif name == "ema":
            return MA_Type.EMA  # 1
        elif name == "wma":
            return MA_Type.WMA  # 2
        elif name == "dema":
            return MA_Type.DEMA  # 3
        elif name == "tema":
            return MA_Type.TEMA  # 4
        elif name == "trima":
            return MA_Type.TRIMA  # 5
        elif name == "kama":
            return MA_Type.KAMA  # 6
        elif name == "mama":
            return MA_Type.MAMA  # 7
        elif name == "t3":
            return MA_Type.T3  # 8
    return 0  # Default: SMA -> 0


def unsigned_differences(
    series: Series, lag: int | None = None, **kwargs
) -> tuple[Series, Series]:
    """Unsigned Differences
    Returns two Series, an unsigned positive and unsigned negative series based
    on the differences of the original series. The positive series are only the
    increases and the negative series are only the decreases.

    Default Example:
    series   = Series([3, 2, 2, 1, 1, 5, 6, 6, 7, 5, 3]) and returns
    positive  = Series([0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    negative = Series([0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1])
    """
    lag = int(lag) if lag is not None else 1
    negative = series.diff(lag)
    negative = negative.fillna(0)
    positive = negative.copy()

    positive[positive <= 0] = 0
    positive[positive > 0] = 1

    negative[negative >= 0] = 0
    negative[negative < 0] = 1

    if kwargs.pop("asint", False):
        positive = positive.astype(int)
        negative = negative.astype(int)

    return positive, negative


def ms2secs(ms, p: int) -> float:
    return round(0.001 * ms, p)


def _speed_group(
    df: DataFrame,
    group: list[str] = [],
    talib: bool = False,
    index_name: str = "Indicator",
    p: int = 4,
) -> list[dict]:
    result = []
    for i in group:
        r = df.ti(i, talib=talib, timed=True)
        if r is None:
            print(f"[S] {i} skipped due to returning None")
            continue  # ti.pivots() sometimes returns None
        ms = float(r.timed.split(" ")[0].split(" ")[0])
        result.append({index_name: i, "ms": ms, "secs": ms2secs(ms, p)})
    return result


def speed_test(
    df: DataFrame,
    only: list[str] | None = None,
    excluded: list[str] | None = None,
    top: int | None = None,
    talib: bool = False,
    ascending: bool = False,
    sortby: str = "secs",
    gradient: bool = False,
    places: int = 5,
    stats: bool = False,
    verbose: bool = False,
    silent: bool = False,
) -> DataFrame | tuple[DataFrame, DataFrame]:
    """Speed Test

    Given a standard ohlcv DataFrame, the Speed Test calculates the
    speed of each indicator of the DataFrame Extension: df.ti.<indicator>().

    Args:
        df (pd.DataFrame): DataFrame with ohlcv columns
        only (list): List of indicators to run. Default: None
        excluded (list): List of indicators to exclude. Default: None
        top (Int): Return a DataFrame the 'top' values. Default: None
        talib (bool): Enable TA Lib. Default: False
        ascending (bool): Ascending Order. Default: False
        sortby (str): Options: "ms", "secs". Default: "secs"
        gradient (bool): Returns a DataFrame the 'top' values with gradient
            styling. Default: False
        places (Int): Decimal places. Default: 5
        stats (bool): Returns a Tuple of two DataFrames. The second tuple
            contains Stats on the performance time. Default: False
        verbose (bool): Display more info. Default: False
        silent (bool): Display nothing. Default: False

    Returns:
        pd.DataFrame: if stats is False
        (pd.DataFrame, pd.DataFrame): if stats is True

    """
    if df.empty:
        print(f"[X] No DataFrame")
        return
    talib = v_bool(talib, False)
    top = int(top) if isinstance(top, int) and top > 0 else None
    stats = v_bool(stats, False)
    verbose = v_bool(verbose, False)
    silent = v_bool(silent, False)

    _ichimoku = ["ichimoku"]
    if excluded is None and isinstance(only, list) and len(only) > 0:
        _indicators = only
    elif only is None and isinstance(excluded, list) and len(excluded) > 0:
        _indicators = df.ti.indicators(as_list=True, exclude=_ichimoku + excluded)
    else:
        _indicators = df.ti.indicators(as_list=True, exclude=_ichimoku)

    if len(_indicators) == 0:
        return None

    _iname = "Indicator"
    if verbose:
        print()
        data = _speed_group(df.copy(), _indicators, talib, _iname, places)
    else:
        _this = StringIO()
        with redirect_stdout(_this):
            data = _speed_group(df.copy(), _indicators, talib, _iname, places)
        _this.close()

    tdf = DataFrame.from_dict(data)
    tdf = tdf.set_index(_iname)
    tdf = tdf.sort_values(by=sortby, ascending=ascending)

    total_timedf = DataFrame(tdf.describe().loc[["min", "50%", "mean", "max"]]).T
    total_timedf["total"] = tdf.sum(axis=0).T
    total_timedf = total_timedf.T

    _div = "=" * 60
    _observations = f"  Observations{'[talib]' if talib else ''}: {df.shape[0]}"
    _quick_slow = "Quickest" if ascending else "Slowest"
    _title = f"  {_quick_slow} Indicators"
    _perfstats = f"Time Stats:\n{total_timedf}"
    if top:
        _title = f"  {_quick_slow} {top} Indicators [{tdf.shape[0]}]"
        tdf = tdf.head(top)

    if not silent:
        print(
            f"\n{_div}\n{_title}\n{_observations}\n{_div}\n{tdf}\n\n{_div}\n{_perfstats}\n\n{_div}\n"
        )

    if isinstance(gradient, bool) and gradient:
        return tdf.style.background_gradient("autumn_r"), total_timedf

    if stats:
        return tdf, total_timedf
    else:
        return tdf
