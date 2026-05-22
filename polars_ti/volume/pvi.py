# -*- coding: utf-8 -*-
from numpy import empty, float64, zeros_like
from numba import njit


@njit(cache=True)
def nb_pvi(np_close, np_volume, initial):
    result = zeros_like(np_close, dtype=float64)
    result[0] = initial

    m = np_close.size
    for i in range(1, m):
        if np_volume[i] > np_volume[i - 1]:
            # Bug fix: was result[i - i] (always 0), now result[i - 1] (previous)
            result[i] = result[i - 1] * (np_close[i] / np_close[i - 1])
        else:
            result[i] = result[i - 1]

    return result


# =============================================================================
# Polars PVI (Positive Volume Index) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


# Reuse existing Numba kernel nb_pvi from above


def pl_pvi(
    close: IntoExpr,
    volume: IntoExpr,
    length: int = 255,
    initial: float = 100.0,
    mamode: str = "ema",
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Positive Volume Index (PVI)

    The Positive Volume Index is a cumulative indicator that uses volume
    change in an attempt to identify where smart money is active.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        length: MA period for signal. Default: 255
        initial: Initial PVI value. Default: 100
        mamode: MA type. Default: 'ema'
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: List of expressions [PVI, PVI_signal]
    """
    from polars_ti.ma import pl_ma

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    _length = length
    _initial = initial
    _mode = mamode.lower()[0] if len(mamode) else ""
    _props = f"{_mode}_{length}"

    def compute_pvi(df: pl.DataFrame) -> pl.Series:
        c = df["close"].to_numpy().astype(np.float64)
        v = df["volume"].to_numpy().astype(np.float64)
        result = nb_pvi(c, v, _initial)
        return pl.Series("PVI", result)

    pvi_expr = pl.struct([close_expr.alias("close"), volume_expr.alias("volume")]).map_batches(
        lambda s: compute_pvi(s.struct.unnest()), return_dtype=pl.Float64
    )

    # Signal = MA(PVI, length)
    pvi_signal_expr = pl_ma(name=mamode, source=pvi_expr, length=length)

    if offset != 0:
        pvi_expr = pvi_expr.shift(offset)
        pvi_signal_expr = pvi_signal_expr.shift(offset)

    return [
        pvi_expr.alias("PVI"),
        pvi_signal_expr.alias(f"PVI{_props}"),
    ]
