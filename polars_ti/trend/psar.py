# -*- coding: utf-8 -*-
# =============================================================================
# Polars PSAR Implementation (Numba kernel)
# =============================================================================
import numpy as np
from numba import njit
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_psar(high, low, sar0, has_close, af0, max_af):
    """Numba kernel for Parabolic SAR."""
    n = len(high)
    sar = np.zeros(n)
    long_arr = np.full(n, np.nan)
    short_arr = np.full(n, np.nan)
    af_arr = np.zeros(n)
    reversal = np.zeros(n, dtype=np.int64)

    # Guard degenerate inputs before touching index 1 (avoids Numba OOB/UB).
    if n < 2:
        if n == 1:
            af_arr[0] = af0
            sar[0] = sar0 if has_close else high[0]
        return long_arr, short_arr, af_arr, reversal

    af = af0
    af_arr[0] = af0
    af_arr[1] = af0

    # Determine initial direction
    falling = (low[0] - low[1]) > (high[1] - high[0]) and (low[0] - low[1]) > 0
    ep = low[0] if falling else high[0]
    if has_close:
        sar[0] = sar0
    else:
        sar[0] = high[0] if falling else low[0]

    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        if falling:
            if low[i] < ep:
                ep = low[i]
                af = min(af + af0, max_af)
            # Guard: clamp to the prior TWO bars' highs (classic 9258bf6). The
            # reversal test must use the GUARDED SAR, not the raw projection, to
            # match TA-Lib (avoids off-by-one misclassification at reversals).
            # max(0, i-2) keeps i==1 from indexing -1 (the last element).
            sar[i] = max(high[i - 1], high[max(0, i - 2)], sar[i])
            reverse = high[i] > sar[i]
        else:
            if high[i] > ep:
                ep = high[i]
                af = min(af + af0, max_af)
            # Symmetric short-stop guard over the prior two bars' lows.
            sar[i] = min(low[i - 1], low[max(0, i - 2)], sar[i])
            reverse = low[i] < sar[i]

        if reverse:
            sar[i] = ep
            af = af0
            falling = not falling
            ep = low[i] if falling else high[i]

        if falling:
            short_arr[i] = sar[i]
        else:
            long_arr[i] = sar[i]

        af_arr[i] = af
        reversal[i] = 1 if reverse else 0

    return long_arr, short_arr, af_arr, reversal


def psar(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr | None = None,
    af0: float | None = None,
    af: float = 0.02,
    max_af: float = 0.2,
    talib: bool = False,
    offset: int = 0,
) -> PlExpr:
    """Polars: Parabolic Stop and Reverse (PSAR)

    Determines trend direction and potential reversals using a trailing
    stop and reverse method.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        af0: Initial acceleration factor. Default: 0.02
        af: Acceleration factor step. Default: 0.02
        max_af: Maximum acceleration factor. Default: 0.2
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with PSARl, PSARs, PSARaf, PSARr columns
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    # Consistency with hl2: a None input expr yields None rather than raising.
    if high_expr is None or low_expr is None:
        return None
    _has_close = close is not None
    close_expr = v_expr(close) if _has_close else low_expr
    # Precedence (matches baseline v_pos_default chain): explicit af0 wins, else
    # the af step, else 0.02. With af0 defaulting to None, `psar(af=X)` is now
    # honored instead of being masked by a truthy af0 default.
    _paf = af if af and af > 0 else 0.02
    _af0 = af0 if af0 and af0 > 0 else _paf
    _use_talib = Imports["talib"] and v_talib(talib)

    def _compute(s: pl.Series) -> pl.Series:
        data = s.struct.unnest()
        h = data["_h"].to_numpy().astype(np.float64)
        l_ = data["_l"].to_numpy().astype(np.float64)
        c_arr = data["_c"].to_numpy().astype(np.float64)
        sar0 = float(c_arr[0]) if _has_close and len(c_arr) > 0 else 0.0
        long_a, short_a, af_a, rev_a = _nb_psar(h, l_, sar0, _has_close, _af0, max_af)

        # The combined SAR line: on the talib path use talib.SAR (a single C call);
        # otherwise the native kernel's SAR (which already equals talib.SAR). Either
        # way, af_a/rev_a come from the native pass — talib.SAR provides no
        # acceleration/reversal, and emitting them as null would leave PSARaf an
        # all-null column, so we keep the (consistent) native values.
        native_combined = np.where(~np.isnan(long_a), long_a, short_a)
        if _use_talib:
            from talib import SAR as _SAR

            combined = _SAR(h, l_, acceleration=_af0, maximum=max_af)
        else:
            combined = native_combined

        # Reclassify long/short from the combined SAR using close (classic
        # 9258bf6): SAR < close -> long, SAR >= close -> short. This matches
        # TA-Lib's convention and avoids off-by-one splits at reversal bars that
        # the `falling` flag alone produces. The combined SAR (the per-bar value)
        # is unchanged; only which of PSARl/PSARs holds it changes.
        if _has_close:
            is_long = combined < c_arr
        else:
            is_long = ~np.isnan(long_a)
        if _has_close or _use_talib:
            long_a = np.where(is_long, combined, np.nan)
            short_a = np.where(~is_long, combined, np.nan)

        if offset != 0:
            for a in [long_a, short_a, af_a]:
                a[:] = np.roll(a, offset)
                if offset > 0:
                    a[:offset] = np.nan
                else:
                    a[offset:] = np.nan
            rev_a = np.roll(rev_a, offset)
            if offset > 0:
                rev_a[:offset] = 0
            else:
                rev_a[offset:] = 0

        _props = f"_{_af0}_{max_af}"
        n = len(h)
        return pl.Series(
            values=[
                {
                    f"PSARl{_props}": long_a[i],
                    f"PSARs{_props}": short_a[i],
                    f"PSARaf{_props}": af_a[i],
                    f"PSARr{_props}": int(rev_a[i]),
                }
                for i in range(n)
            ]
        )

    _props = f"_{_af0}_{max_af}"
    fields = [
        pl.Field(f"PSARl{_props}", pl.Float64),
        pl.Field(f"PSARs{_props}", pl.Float64),
        pl.Field(f"PSARaf{_props}", pl.Float64),
        pl.Field(f"PSARr{_props}", pl.Int64),
    ]
    return (
        pl.struct(
            high_expr.alias("_h"),
            low_expr.alias("_l"),
            close_expr.alias("_c"),
        )
        .map_batches(_compute, return_dtype=pl.Struct(fields))
        .alias(f"PSAR{_props}")
    )
