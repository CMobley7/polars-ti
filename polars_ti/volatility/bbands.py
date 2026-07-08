# -*- coding: utf-8 -*-
# =============================================================================
# Polars BBANDS Implementation (Pure Native Polars + TA-Lib option)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.maps import Imports
from polars_ti.utils._validate import v_expr
from polars_ti.utils import v_talib


def bbands(
    close: IntoExpr,
    length: int = 5,
    std: float = 2.0,
    ddof: int = 0,
    talib: bool = True,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Bollinger Bands (BBANDS)

    Uses TA-Lib when available and talib=True, otherwise pure Polars expressions.

    Sources:
        https://www.tradingview.com/wiki/Bollinger_Bands_(BB)

    Args:
        close: Column name or pl.Expr for 'close'
        length: SMA period. Default: 5
        std: Standard deviation multiplier. Default: 2.0
        ddof: Delta degrees of freedom (ignored when talib=True). Default: 1
        talib: If True and TA-Lib installed, uses TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with BBL, BBM, BBU, BBB, BBP columns
    """
    close_expr = v_expr(close)

    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib)
    _length = length
    _std = std
    _ddof = ddof
    _offset = offset

    if _use_talib:
        # TA-Lib path
        def compute_bbands_talib(s: pl.Series) -> pl.Series:
            from talib import BBANDS

            arr = s.to_numpy().astype(np.float64)
            upper, mid, lower = BBANDS(arr, _length, _std, _std, 0)  # 0 = SMA

            # Bandwidth and percent
            ulr = upper - lower
            bandwidth = np.full_like(ulr, np.nan, dtype=np.float64)
            percent = np.full_like(ulr, np.nan, dtype=np.float64)
            np.divide(100 * ulr, mid, out=bandwidth, where=mid != 0)
            np.divide(arr - lower, ulr, out=percent, where=ulr != 0)

            if _offset != 0:
                lower = np.roll(lower, _offset)
                mid = np.roll(mid, _offset)
                upper = np.roll(upper, _offset)
                bandwidth = np.roll(bandwidth, _offset)
                percent = np.roll(percent, _offset)
                if _offset > 0:
                    lower[:_offset] = np.nan
                    mid[:_offset] = np.nan
                    upper[:_offset] = np.nan
                    bandwidth[:_offset] = np.nan
                    percent[:_offset] = np.nan
                else:
                    lower[_offset:] = np.nan
                    mid[_offset:] = np.nan
                    upper[_offset:] = np.nan
                    bandwidth[_offset:] = np.nan
                    percent[_offset:] = np.nan

            _props = f"_{_length}_{_std}"
            return pl.DataFrame(
                {
                    f"BBL{_props}": lower,
                    f"BBM{_props}": mid,
                    f"BBU{_props}": upper,
                    f"BBB{_props}": bandwidth,
                    f"BBP{_props}": percent,
                }
            ).to_struct(f"BBANDS{_props}")

        _props = f"_{length}_{std}"
        return close_expr.map_batches(
            compute_bbands_talib,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"BBL{_props}", pl.Float64),
                    pl.Field(f"BBM{_props}", pl.Float64),
                    pl.Field(f"BBU{_props}", pl.Float64),
                    pl.Field(f"BBB{_props}", pl.Float64),
                    pl.Field(f"BBP{_props}", pl.Float64),
                ]
            ),
        ).alias(f"BBANDS{_props}")
    else:
        # Pure Polars path with pl_sma composition
        from polars_ti.overlap.sma import sma

        mid = sma(close_expr, length=length)
        std_dev = close_expr.rolling_std(window_size=length, min_samples=length, ddof=ddof)

        deviations = pl.lit(std) * std_dev
        lower = mid - deviations
        upper = mid + deviations

        ulr = upper - lower
        # Guard divide-by-zero to match the TA-Lib branch (NaN, not inf) when
        # mid==0 (e.g. zero-mean/detrended input) or ulr==0 (flat bands).
        bandwidth = pl.when(mid != 0).then((pl.lit(100.0) * ulr) / mid).otherwise(None)
        percent = pl.when(ulr != 0).then((close_expr - lower) / ulr).otherwise(None)

        if offset != 0:
            lower = lower.shift(offset)
            mid = mid.shift(offset)
            upper = upper.shift(offset)
            bandwidth = bandwidth.shift(offset)
            percent = percent.shift(offset)

        _props = f"_{length}_{std}"

        return pl.struct(
            [
                lower.alias(f"BBL{_props}"),
                mid.alias(f"BBM{_props}"),
                upper.alias(f"BBU{_props}"),
                bandwidth.alias(f"BBB{_props}"),
                percent.alias(f"BBP{_props}"),
            ]
        ).alias(f"BBANDS{_props}")
