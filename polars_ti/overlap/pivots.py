# -*- coding: utf-8 -*-
from numba import njit

from polars_ti.utils._core import nb_non_zero_range


@njit(cache=True)
def pivot_camarilla(high, low, close):
    tp = (high + low + close) / 3
    hl_range = nb_non_zero_range(high, low)

    s1 = close - 11 / 120 * hl_range
    s2 = close - 11 / 60 * hl_range
    s3 = close - 0.275 * hl_range
    s4 = close - 0.55 * hl_range

    r1 = close + 11 / 120 * hl_range
    r2 = close + 11 / 60 * hl_range
    r3 = close + 0.275 * hl_range
    r4 = close + 0.55 * hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@njit(cache=True)
def pivot_classic(high, low, close):
    tp = (high + low + close) / 3
    hl_range = nb_non_zero_range(high, low)

    s1 = 2 * tp - high
    s2 = tp - hl_range
    s3 = tp - 2 * hl_range
    s4 = tp - 3 * hl_range

    r1 = 2 * tp - low
    r2 = tp + hl_range
    r3 = tp + 2 * hl_range
    r4 = tp + 3 * hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@njit(cache=True)
def pivot_demark(open_, high, low, close):
    # Per-bar branch selection (not a whole-series .all() collapse):
    #   down bar (close < open):  0.25 * (high + 2*low + close)
    #   up bar   (close > open):  0.25 * (2*high + low + close)
    #   equal:                    0.25 * (high + low + 2*close)
    tp = np.where(
        close < open_,
        0.25 * (high + 2.0 * low + close),
        np.where(
            close > open_,
            0.25 * (2.0 * high + low + close),
            0.25 * (high + low + 2.0 * close),
        ),
    )

    s1 = 2 * tp - high
    r1 = 2 * tp - low

    return tp, s1, r1


@njit(cache=True)
def pivot_fibonacci(high, low, close):
    tp = (high + low + close) / 3
    hl_range = nb_non_zero_range(high, low)

    s1 = tp - 0.382 * hl_range
    s2 = tp - 0.618 * hl_range
    s3 = tp - hl_range

    r1 = tp + 0.382 * hl_range
    r2 = tp + 0.618 * hl_range
    r3 = tp + hl_range

    return tp, s1, s2, s3, r1, r2, r3


@njit(cache=True)
def pivot_traditional(high, low, close):
    tp = (high + low + close) / 3
    hl_range = nb_non_zero_range(high, low)

    s1 = 2 * tp - high
    s2 = tp - hl_range
    s3 = tp - 2 * hl_range
    s4 = tp - 2 * hl_range

    r1 = 2 * tp - low
    r2 = tp + hl_range
    r3 = tp + 2 * hl_range
    r4 = tp + 2 * hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@njit(cache=True)
def pivot_woodie(open_, high, low):
    tp = (2 * open_ + high + low) / 4
    hl_range = nb_non_zero_range(high, low)

    s1 = 2 * tp - high
    s2 = tp - hl_range
    s3 = low - 2 * (high - tp)
    s4 = s3 - hl_range

    r1 = 2 * tp - low
    r2 = tp + hl_range
    r3 = high + 2 * (tp - low)
    r4 = r3 + hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


# =============================================================================
# Polars PIVOTS Implementation (Pure Polars + Numba)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr


def pivots(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    open_: IntoExpr | None = None,
    method: str = "traditional",
    anchor: str = "D",
) -> pl.Expr:
    """Polars: Pivot Points (Pure Polars + Numba)

    Calculates support and resistance levels from previous price action.
    Each bar's pivot is computed from that bar's OHLC and then shifted forward
    by one period (the pivot applies to the *next* period), matching the old
    pandas-ti behaviour on a daily anchor.

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        open_: Column name or pl.Expr for 'open' (required for woodie/demark).
        method: Pivot calculation method. Default: "traditional"
            Options: traditional, fibonacci, woodie, classic, demark, camarilla
        anchor: Anchor frequency. Only the per-bar daily anchor ("D") is
            supported through the expression API. Default: "D"

    Returns:
        pl.Expr: Struct expression with pivot columns (P, S1-S4, R1-R4)
    """
    method = method.lower()
    anchor = anchor.upper()
    _props = f"PIVOTS_{method[:4].upper()}_{anchor}"

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    if open_ is not None:
        open_expr = v_expr(open_)
    elif method in ("woodie", "demark"):
        raise ValueError(f"pivots(method={method!r}) requires an 'open_' series")
    else:
        # Placeholder; unused for methods that don't need open.
        open_expr = close_expr

    def compute_pivots(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        np_open = df["_open"].to_numpy().astype(np.float64)
        np_high = df["_high"].to_numpy().astype(np.float64)
        np_low = df["_low"].to_numpy().astype(np.float64)
        np_close = df["_close"].to_numpy().astype(np.float64)

        n = len(np_close)
        nan_array = np.full(n, np.nan)

        if method == "camarilla":
            tp, s1, s2, s3, s4, r1, r2, r3, r4 = pivot_camarilla(np_high, np_low, np_close)
        elif method == "classic":
            tp, s1, s2, s3, s4, r1, r2, r3, r4 = pivot_classic(np_high, np_low, np_close)
        elif method == "demark":
            tp, s1, r1 = pivot_demark(np_open, np_high, np_low, np_close)
            s2 = s3 = s4 = r2 = r3 = r4 = nan_array
        elif method == "fibonacci":
            tp, s1, s2, s3, r1, r2, r3 = pivot_fibonacci(np_high, np_low, np_close)
            s4, r4 = nan_array, nan_array
        elif method == "woodie":
            tp, s1, s2, s3, s4, r1, r2, r3, r4 = pivot_woodie(np_open, np_high, np_low)
        else:  # Traditional
            tp, s1, s2, s3, s4, r1, r2, r3, r4 = pivot_traditional(np_high, np_low, np_close)

        cols = {
            f"{_props}_P": tp,
            f"{_props}_S1": s1,
            f"{_props}_S2": s2,
            f"{_props}_S3": s3,
            f"{_props}_S4": s4,
            f"{_props}_R1": r1,
            f"{_props}_R2": r2,
            f"{_props}_R3": r3,
            f"{_props}_R4": r4,
        }

        # The pivot applies to the *next* period: shift each column forward by 1.
        shifted = {}
        for name, arr in cols.items():
            out = np.full(n, np.nan)
            out[1:] = arr[:-1]
            shifted[name] = out

        return pl.Series([{name: shifted[name][i] for name in shifted} for i in range(n)])

    names = ["P", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4"]
    fields = [pl.Field(f"{_props}_{n}", pl.Float64) for n in names]

    return (
        pl.struct(
            [
                open_expr.alias("_open"),
                high_expr.alias("_high"),
                low_expr.alias("_low"),
                close_expr.alias("_close"),
            ]
        )
        .map_batches(compute_pivots, return_dtype=pl.Struct(fields))
        .alias(_props)
    )
