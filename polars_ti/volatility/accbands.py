# -*- coding: utf-8 -*-
# =============================================================================
# Polars ACCBANDS Implementation (pl_ma composition + TA-Lib option)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.maps import Imports
from polars_ti.utils._math import non_zero_range
from polars_ti.utils._validate import v_expr
from polars_ti.utils import v_talib


def accbands(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 20,
    c: float = 4.0,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Acceleration Bands (ACCBANDS)

    Plots upper and lower envelope bands around a moving average.

    Uses TA-Lib when available and talib=True, otherwise uses pl_ma composition.

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/acceleration-bands-abands/

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        length: MA period. Default: 20
        c: Multiplier. Default: 4.0
        mamode: MA type ('sma', 'ema', etc.). Default: 'sma'
        talib: If True and TA-Lib installed, uses TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with ACCBL, ACCBM, ACCBU columns
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib)
    _length = length
    _c = c
    _mamode = mamode.lower() if isinstance(mamode, str) else "sma"
    _offset = offset

    if _use_talib:
        # TA-Lib path: use map_batches for direct TA-Lib ACCBANDS call
        def compute_accbands_talib(struct: pl.Series) -> pl.Series:
            from talib import ACCBANDS

            df = struct.struct.unnest()
            high_arr = df["_high"].to_numpy().astype(np.float64)
            low_arr = df["_low"].to_numpy().astype(np.float64)
            close_arr = df["_close"].to_numpy().astype(np.float64)

            upper, mid, lower = ACCBANDS(high_arr, low_arr, close_arr, timeperiod=_length)

            if _offset != 0:
                lower = np.roll(lower, _offset)
                mid = np.roll(mid, _offset)
                upper = np.roll(upper, _offset)
                if _offset > 0:
                    lower[:_offset] = np.nan
                    mid[:_offset] = np.nan
                    upper[:_offset] = np.nan

            return pl.DataFrame(
                {
                    f"ACCBL_{_length}": lower,
                    f"ACCBM_{_length}": mid,
                    f"ACCBU_{_length}": upper,
                }
            ).to_struct(f"ACCBANDS_{_length}")

        return (
            pl.struct(
                [
                    high_expr.alias("_high"),
                    low_expr.alias("_low"),
                    close_expr.alias("_close"),
                ]
            )
            .map_batches(
                compute_accbands_talib,
                return_dtype=pl.Struct(
                    [
                        pl.Field(f"ACCBL_{length}", pl.Float64),
                        pl.Field(f"ACCBM_{length}", pl.Float64),
                        pl.Field(f"ACCBU_{length}", pl.Float64),
                    ]
                ),
            )
            .alias(f"ACCBANDS_{length}")
        )
    else:
        # Polars composition path using pl_ma
        from polars_ti.ma import ma

        # High-Low range with non-zero protection
        hl_range = non_zero_range(high_expr, low_expr)

        # hl_ratio = c * hl_range / (high + low)
        hl_ratio = (pl.lit(c) * hl_range) / (high_expr + low_expr)

        # Lower = low * (1 - hl_ratio), Upper = high * (1 + hl_ratio)
        lower_raw = low_expr * (pl.lit(1.0) - hl_ratio)
        upper_raw = high_expr * (pl.lit(1.0) + hl_ratio)

        # Apply MA to each using pl_ma dispatcher
        lower = ma(name=mamode, source=lower_raw, length=length, talib=False)
        mid = ma(name=mamode, source=close_expr, length=length, talib=False)
        upper = ma(name=mamode, source=upper_raw, length=length, talib=False)

        if offset != 0:
            lower = lower.shift(offset)
            mid = mid.shift(offset)
            upper = upper.shift(offset)

        return pl.struct(
            [
                lower.alias(f"ACCBL_{length}"),
                mid.alias(f"ACCBM_{length}"),
                upper.alias(f"ACCBU_{length}"),
            ]
        ).alias(f"ACCBANDS_{length}")
