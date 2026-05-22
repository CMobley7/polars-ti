# -*- coding: utf-8 -*-
# =============================================================================
# Polars OTT Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.ma import pl_ma


@njit(cache=True)
def nb_ott(mavg: np.ndarray, multiplier: float) -> tuple:
    """Numba-optimized OTT calculation from MA values."""
    m = len(mavg)
    dir_ = np.ones(m, dtype=np.int64)
    trend = np.zeros(m, dtype=np.float64)
    after_dir = np.ones(m, dtype=np.int64)

    # Calculate bands as percentage of moving average
    matr = multiplier * mavg * 0.01
    upperband = mavg + matr
    lowerband = mavg - matr

    # Make copies for modification
    ub = upperband.copy()
    lb = lowerband.copy()

    for i in range(1, m):
        # Determine direction
        if mavg[i] > ub[i - 1]:
            dir_[i] = 1
        elif mavg[i] < lb[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            # Preserve bands in direction
            if dir_[i] > 0 and lb[i] < lb[i - 1]:
                lb[i] = lb[i - 1]
            if dir_[i] < 0 and ub[i] > ub[i - 1]:
                ub[i] = ub[i - 1]

        # Calculate OTT trend line
        if dir_[i] > 0:
            trend[i] = lb[i] * (200 + multiplier) / 200
        else:
            trend[i] = ub[i] * (200 - multiplier) / 200

    # Calculate after direction
    for i in range(2, m):
        if mavg[i] > trend[i - 2]:
            after_dir[i] = 1
        elif mavg[i] < trend[i - 1]:
            after_dir[i] = -1
        else:
            after_dir[i] = after_dir[i - 1]

    return trend, after_dir


def pl_ott(
    close: IntoExpr,
    length: int = 5,
    multiplier: float = 2.4,
    mamode: str = "vidya",
    offset: int = 0,
) -> pl.Expr:
    """Polars: Optimized Trend Tracker (OTT)

    Returns a struct with OTT, OTTSL, OTTd fields.

    Args:
        close: Column name or pl.Expr for 'close'
        length: MA period. Default: 5
        multiplier: Band width percentage. Default: 2.4
        mamode: MA type. Default: "vidya"
        offset: Offset periods. Default: 0

    Returns:
        pl.Expr: Struct with OTT, OTTSL, OTTd fields
    """
    close_expr = v_expr(close)
    _props = f"_{length}_{multiplier}"

    # Get the MA expression
    ma_expr = pl_ma(mamode, close, length=length).alias(f"OTTSL{_props}")

    def compute_ott(struct: pl.Series) -> pl.Series:
        # Extract MA values from struct
        df = struct.struct.unnest()
        mavg = df[f"OTTSL{_props}"].to_numpy().astype(np.float64)

        trend, after_dir = nb_ott(mavg, multiplier)

        if offset != 0:
            trend = np.roll(trend, offset)
            after_dir = np.roll(after_dir, offset)
            if offset > 0:
                trend[:offset] = np.nan
                after_dir[:offset] = 1

        return pl.Series(
            [
                {
                    f"OTT{_props}": trend[i],
                    f"OTTSL{_props}": mavg[i],
                    f"OTTd{_props}": int(after_dir[i]),
                }
                for i in range(len(mavg))
            ]
        )

    # Build struct with MA first, then compute OTT from it
    return pl.struct([ma_expr]).map_batches(compute_ott).alias(f"OTT{_props}")
