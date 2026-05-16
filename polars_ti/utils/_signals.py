# -*- coding: utf-8 -*-
import polars as pl

from polars_ti._typing import DictLike, Int, IntFloat
from polars_ti.utils._validate import v_offset, v_series


def _as_series(value, name: str | None = None, length: int | None = None) -> pl.Series:
    if isinstance(value, pl.Series):
        return value.rename(name) if name else value
    if isinstance(value, (int, float, complex)):
        if length is None:
            raise ValueError("length is required for scalar signal comparisons")
        return pl.Series(name or f"{value}".replace(".", "_"), [value] * length)
    return pl.Series(name or getattr(value, "name", "series"), value)


def _set_name(series: pl.Series, name: str) -> pl.Series:
    return series.rename(name)


def _above_below(
    series_a,
    series_b,
    above: bool = True,
    asint: bool = True,
    offset: Int = None,
    **kwargs,
):
    # Verify
    series_a = _as_series(v_series(series_a))
    series_b = _as_series(v_series(series_b))
    offset = v_offset(offset)

    # Calculate
    if above:
        current = series_a >= series_b
    else:
        current = series_a <= series_b

    if asint:
        current = current.cast(pl.Int64)

    # Offset
    if offset != 0:
        current = current.shift(offset)

    # Name and Category
    return _set_name(current, f"{series_a.name}_{'A' if above else 'B'}_{series_b.name}")


def above(
    series_a, series_b, asint: bool = True, offset: Int = None, **kwargs
):
    return _above_below(
        series_a, series_b, above=True, asint=asint, offset=offset, **kwargs
    )


def above_value(
    series_a, value: IntFloat, asint: bool = True, offset: Int = None, **kwargs
):
    if not isinstance(value, (int, float, complex)):
        print("[X] value is not a number")
        return
    series_b = _as_series(value, name=f"{value}".replace(".", "_"), length=len(series_a))

    return _above_below(
        series_a, series_b, above=True, asint=asint, offset=offset, **kwargs
    )


def below(
    series_a, series_b, asint: bool = True, offset: Int = None, **kwargs
):
    return _above_below(
        series_a, series_b, above=False, asint=asint, offset=offset, **kwargs
    )


def below_value(
    series_a, value: IntFloat, asint: bool = True, offset: Int = None, **kwargs
):
    if not isinstance(value, (int, float, complex)):
        print("[X] value is not a number")
        return
    series_b = _as_series(value, name=f"{value}".replace(".", "_"), length=len(series_a))
    return _above_below(
        series_a, series_b, above=False, asint=asint, offset=offset, **kwargs
    )


def cross_value(
    series_a,
    value: IntFloat,
    above: bool = True,
    equal: bool = True,
    asint: bool = True,
    offset: Int = None,
    **kwargs,
):
    series_b = _as_series(value, name=f"{value}".replace(".", "_"), length=len(series_a))

    return cross(series_a, series_b, above, equal, asint, offset, **kwargs)


def cross(
    series_a,
    series_b,
    above: bool = True,
    equal: bool = True,
    asint: bool = True,
    offset: Int = None,
    **kwargs: DictLike,
):
    # Validate
    series_a = _as_series(v_series(series_a))
    series_b = _as_series(v_series(series_b))
    offset = v_offset(offset)

    # Calculate
    if above:
        current = series_a >= series_b if equal else series_a > series_b
        previous = series_a.shift(1) < series_b.shift(1)
    else:
        current = series_a <= series_b if equal else series_a < series_b
        previous = series_a.shift(1) > series_b.shift(1)

    cross = current & previous
    # ensure there is no cross on the first entry
    cross = pl.Series(cross.name, [False] + cross.slice(1).to_list())

    if asint:
        cross = cross.cast(pl.Int64)

    # Offset
    if offset != 0:
        cross = cross.shift(offset)

    # Name and Category
    return _set_name(cross, f"{series_a.name}_{'XA' if above else 'XB'}_{series_b.name}")


def signals(
    indicator,
    xa: IntFloat,
    xb: IntFloat,
    cross_values: bool,
    xserie,
    xserie_a,
    xserie_b,
    cross_series: bool,
    offset: Int,
):
    df = pl.DataFrame()
    if xa is not None and isinstance(xa, (int, float)):
        if cross_values:
            crossed_above_start = cross_value(indicator, xa, above=True, offset=offset)
            crossed_above_end = cross_value(indicator, xa, above=False, offset=offset)
            df = df.with_columns(crossed_above_start, crossed_above_end)
        else:
            crossed_above = above_value(indicator, xa, offset=offset)
            df = df.with_columns(crossed_above)

    if xb is not None and isinstance(xb, (int, float)):
        if cross_values:
            crossed_below_start = cross_value(indicator, xb, above=True, offset=offset)
            crossed_below_end = cross_value(indicator, xb, above=False, offset=offset)
            df = df.with_columns(crossed_below_start, crossed_below_end)
        else:
            crossed_below = below_value(indicator, xb, offset=offset)
            df = df.with_columns(crossed_below)

    # xseries is the default value for both xserie_a and xserie_b
    if xserie_a is None:
        xserie_a = xserie
    if xserie_b is None:
        xserie_b = xserie

    if xserie_a is not None and v_series(xserie_a):
        if cross_series:
            cross_serie_above = cross(indicator, xserie_a, above=True, offset=offset)
        else:
            cross_serie_above = above(indicator, xserie_a, offset=offset)

        df = df.with_columns(cross_serie_above)

    if xserie_b is not None and v_series(xserie_b):
        if cross_series:
            cross_serie_below = cross(indicator, xserie_b, above=False, offset=offset)
        else:
            cross_serie_below = below(indicator, xserie_b, offset=offset)

        df = df.with_columns(cross_serie_below)

    return df
