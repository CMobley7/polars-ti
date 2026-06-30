# -*- coding: utf-8 -*-
# =============================================================================
# Polars VFI (Volume Flow Indicator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._math import non_zero_range
from polars_ti.utils._validate import v_expr


def vfi(
    close: IntoExpr,
    volume: IntoExpr,
    length: int = 130,
    coef: float = 0.2,
    vcoef: float = 2.5,
    mamode: str = "ema",
    offset: int = 0,
) -> PlExpr:
    """Polars: Volume Flow Indicator (VFI)

    Combines price movement with volume to show money flow.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        length: Period. Default: 130
        coef: Volatility threshold coefficient. Default: 0.2
        vcoef: Volume coefficient. Default: 2.5
        mamode: MA type for smoothing. Default: 'ema'
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: VFI expression
    """
    from polars_ti.ma import ma

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    # Volume average and clipped volume
    vave = volume_expr.rolling_mean(window_size=length, min_samples=length).shift(1)
    vmax = vave * vcoef

    # min(volume, vmax)
    vc = pl.min_horizontal(volume_expr, vmax)

    # Money flow with volatility threshold
    inter = close_expr.diff(1)
    cutoff = coef * close_expr
    mf = pl.when(inter.abs() > cutoff).then(inter).otherwise(0.0)

    # Volume-weighted money flow
    vcp = vc * mf

    # VFI = sum(vcp, length) / rolling_mean(vave, length)
    vave_mean = vave.rolling_mean(window_size=length, min_samples=length)
    # Protect against division by zero using shared utility
    vave_mean_safe = non_zero_range(vave_mean, pl.lit(0.0))

    vfi_expr = vcp.rolling_sum(window_size=length, min_samples=length) / vave_mean_safe

    # Smooth with EMA(3)
    vfi_expr = ma(name=mamode, source=vfi_expr, length=3)

    if offset != 0:
        vfi_expr = vfi_expr.shift(offset)

    return vfi_expr.alias(f"VFI_{length}")
