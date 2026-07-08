# -*- coding: utf-8 -*-
from datetime import datetime, timezone
from time import localtime, perf_counter

from polars_ti._typing import Float, Optional, Tuple
from polars_ti.maps import EXCHANGE_TZ


def df_dates(df, dates: Tuple[str, list] = None):
    """Yields the DataFrame with the given dates"""
    if dates is None:
        return None
    if not isinstance(dates, list):
        dates = [dates]
    if hasattr(df, "filter") and hasattr(df, "columns"):
        date_col = "date" if "date" in df.columns else df.columns[0]
        return df.filter(pl.col(date_col).cast(str).is_in(dates))
    return df[df.index.isin(dates)]


def df_month_to_date(df):
    """Yields the Month-to-Date (MTD) DataFrame"""
    now = datetime.now()
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if hasattr(df, "filter") and hasattr(df, "columns"):
        date_col = "date" if "date" in df.columns else df.columns[0]
        return df.filter(pl.col(date_col) >= start)
    try:
        in_mtd = df.index >= start.strftime("%Y-%m-%d")
        if any(in_mtd):
            return df[in_mtd]
    except AttributeError:
        pass
    return df


def df_quarter_to_date(df):
    """Yields the Quarter-to-Date (QTD) DataFrame"""
    now = datetime.now()
    quarter_start_month = ((now.month - 1) // 3) * 3 + 1
    start = now.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    if hasattr(df, "filter") and hasattr(df, "columns"):
        date_col = "date" if "date" in df.columns else df.columns[0]
        return df.filter(pl.col(date_col) >= start)
    try:
        in_qtr = df.index >= start.strftime("%Y-%m-%d")
        if any(in_qtr):
            return df[in_qtr]
    except AttributeError:
        pass
    return df


def df_year_to_date(df):
    """Yields the Year-to-Date (YTD) DataFrame"""
    start = datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    if hasattr(df, "filter") and hasattr(df, "columns"):
        date_col = "date" if "date" in df.columns else df.columns[0]
        return df.filter(pl.col(date_col) >= start)
    try:
        in_ytd = df.index >= start.strftime("%Y-%m-%d")
        if any(in_ytd):
            return df[in_ytd]
    except AttributeError:
        pass
    return df


def final_time(stime: Float) -> str:
    """Human readable elapsed time. Calculates the final time elapsed since
    stime and returns a string with microseconds and seconds."""
    time_diff = perf_counter() - stime
    return f"{time_diff * 1000:2.4f} ms ({time_diff:2.4f} s)"


_DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
_MONTH_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def get_time(exchange: str = "NYSE", full: bool = True, to_string: bool = False) -> Optional[str]:
    """Returns Current Time, Day of the Year and Percentage, and the current
    time of the selected Exchange."""
    try:
        import datetime as _dt

        tz = EXCHANGE_TZ["NYSE"]  # Default is NYSE (Eastern Time Zone)
        if isinstance(exchange, str):
            exchange = exchange.upper()
            tz = EXCHANGE_TZ.get(exchange, EXCHANGE_TZ["NYSE"])

        today = _dt.datetime.now()
        day_name = _DAY_NAMES[today.weekday()]
        month_name = _MONTH_NAMES[today.month]
        date = f"{day_name} {month_name} {today.day}, {today.year}"

        exchange_time = f"{(today.hour + tz) % 24}:{today.minute:02d}:{today.second:02d}"

        # Day of year (1-indexed)
        doy_num = today.timetuple().tm_yday

        if full:
            lt = localtime()
            local_ = f"Local: {lt.tm_hour}:{lt.tm_min:02d}:{lt.tm_sec:02d} {lt.tm_zone}"
            doy = f"Day {doy_num}/365 ({100 * round(doy_num / 365, 2):.2f}%)"
            exchange_ = f"{exchange}: {exchange_time}"
            s = f"{date}, {exchange_}, {local_}, {doy}"
        else:
            s = f"{date}, {exchange}: {exchange_time}"

        return s if to_string else print(s)
    except Exception as exc:
        return f"[get_time error: {exc}]" if to_string else None


def to_utc(df):
    """Either localizes the DataFrame Index to UTC or it applies tz_convert to
    set the Index to UTC.
    """
    if hasattr(df, "with_columns") and hasattr(df, "columns"):
        date_cols = [name for name, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Date, pl.Datetime)]
        if not date_cols:
            return df
        return df.with_columns(pl.col(date_cols[0]).dt.replace_time_zone("UTC"))
    if not df.empty:
        try:
            df.index = df.index.tz_localize("UTC")
        except TypeError:
            df.index = df.index.tz_convert("UTC")
    return df


def unix_convert(ts):
    """
    Converts timestamps from polygon to readable datetime strings.

    :param ts: The timestamp(s). An integer posix timestamp.
    :return: The converted datetime string
    """
    if isinstance(ts, (list, tuple)):
        return [datetime.fromtimestamp(x / 1000, tz=timezone.utc) for x in ts]
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


# Aliases
mtd = df_month_to_date
qtd = df_quarter_to_date
ytd = df_year_to_date


# =============================================================================
# Polars Time Utilities (for Polars-TI conversion)
# =============================================================================
import polars as pl

from polars_ti._typing import PlExpr, PolarsFrame


def total_time(df: pl.DataFrame, time_col: str, tf: str = "years") -> float:
    """Polars: Calculates the total time span of a DataFrame.

    Args:
        df: Polars DataFrame with datetime column
        time_col: Name of the datetime column
        tf: Time frame - 'years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds'

    Returns:
        Total time in the specified unit
    """
    first = df[time_col].min()
    last = df[time_col].max()

    if first is None or last is None:
        return 0.0

    time_diff = last - first
    total_seconds = time_diff.total_seconds()
    days = total_seconds / 86400

    TimeFrame = {
        "years": days / 365.242199074074074,
        "months": days / 30.417,
        "weeks": days / 7,
        "days": days,
        "hours": total_seconds / 3600,
        "minutes": total_seconds / 60,
        "seconds": total_seconds,
    }

    return TimeFrame.get(tf, TimeFrame["years"])


def filter_dates(df: pl.DataFrame, time_col: str, dates: list[str]) -> pl.DataFrame:
    """Polars: Filter DataFrame to specific dates."""
    return df.filter(pl.col(time_col).dt.date().cast(str).is_in(dates))


def year_to_date(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    """Polars: Filter to Year-to-Date records."""
    from datetime import datetime

    start_of_year = datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return df.filter(pl.col(time_col) >= start_of_year)


def month_to_date(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    """Polars: Filter to Month-to-Date records."""
    from datetime import datetime

    start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return df.filter(pl.col(time_col) >= start_of_month)
