# -*- coding: utf-8 -*-
# =============================================================================
# Polars RVI Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.ma import pl_ma
from polars_ti.utils._validate import v_expr


def _pl_rvi_single(arr: np.ndarray, length: int, scalar: float, mamode: str, drift: int) -> np.ndarray:
    """Core RVI computation on a NumPy array.

    Computes RVI using rolling std split by up/down direction, then
    EMA-smooths both sides and calculates the RSI-style ratio.
    """
    n = len(arr)
    diff = np.diff(arr, prepend=np.nan)

    # Rolling STD with window = length
    std_arr = np.full(n, np.nan)
    for i in range(length - 1, n):
        window = arr[i - length + 1 : i + 1]
        if not np.any(np.isnan(window)):
            std_arr[i] = np.std(window, ddof=0)

    # Separate up/down std
    up_std = np.where(diff > 0, std_arr, 0.0)
    dn_std = np.where(diff <= 0, std_arr, 0.0)

    # Smooth via pl_ma through temporary DataFrame
    tmp = pl.DataFrame({"_up": up_std.astype(np.float64), "_dn": dn_std.astype(np.float64)})
    ma_expr_up = pl_ma(mamode, "_up", length=length)
    ma_expr_dn = pl_ma(mamode, "_dn", length=length)
    up_smooth = tmp.select(ma_expr_up).to_series().to_numpy()
    dn_smooth = tmp.select(ma_expr_dn).to_series().to_numpy()

    # RVI = scalar * up / (up + dn)
    denom = up_smooth + dn_smooth
    denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return scalar * up_smooth / denom_safe


def pl_rvi(
    close: IntoExpr,
    high: IntoExpr | None = None,
    low: IntoExpr | None = None,
    length: int = 14,
    scalar: float = 100.0,
    refined: bool = False,
    thirds: bool = False,
    mamode: str = "ema",
    drift: int = 1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Relative Volatility Index (RVI)

    RVI adds up standard deviations based on price direction (unlike RSI
    which adds up price changes).

    Args:
        close: Column name or pl.Expr for 'close'
        high: Column name or pl.Expr for 'high' (for refined/thirds mode)
        low: Column name or pl.Expr for 'low' (for refined/thirds mode)
        length: The period. Default: 14
        scalar: Scale factor. Default: 100.0
        refined: Average of RVI(high) and RVI(low). Default: False
        thirds: Average of high, low, and close. Default: False
        mamode: MA type. Default: 'ema'
        drift: The diff period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: RVI expression
    """
    close_expr = v_expr(close)
    _length = length
    _scalar = scalar
    _drift = drift
    _mamode = mamode
    _refined = refined
    _thirds = thirds

    if refined or thirds:
        high_expr = v_expr(high)
        low_expr = v_expr(low)

        def compute_refined(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            c_arr = df["close"].to_numpy().astype(np.float64)
            h_arr = df["high"].to_numpy().astype(np.float64)
            l_arr = df["low"].to_numpy().astype(np.float64)

            high_rvi = _pl_rvi_single(h_arr, _length, _scalar, _mamode, _drift)
            low_rvi = _pl_rvi_single(l_arr, _length, _scalar, _mamode, _drift)

            if _thirds:
                close_rvi = _pl_rvi_single(c_arr, _length, _scalar, _mamode, _drift)
                result = (high_rvi + low_rvi + close_rvi) / 3.0
            else:
                result = 0.5 * (high_rvi + low_rvi)

            return pl.Series(result)

        struct_expr = pl.struct(close=close_expr, high=high_expr, low=low_expr)
        result = struct_expr.map_batches(compute_refined, return_dtype=pl.Float64)
        _mode = "r" if refined else "t"
    else:

        def compute_simple(s: pl.Series) -> pl.Series:
            arr = s.to_numpy().astype(np.float64)
            return pl.Series(_pl_rvi_single(arr, _length, _scalar, _mamode, _drift))

        result = close_expr.map_batches(compute_simple, return_dtype=pl.Float64)
        _mode = ""

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"RVI{_mode}_{length}")
