# -*- coding: utf-8 -*-
# =============================================================================
# Polars VHM (Volume Heatmap) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_vhm(
    volume: IntoExpr,
    length: int = 610,
    slength: int | None = None,
    mamode: str = "sma",
    offset: int = 0,
) -> PlExpr:
    """Polars: Volume Heatmap (VHM)

    Indicates market/trend strength based on volume deviation.

    Args:
        volume: Column name or pl.Expr for 'volume'
        length: Mean calculation period. Default: 610
        slength: StdDev calculation period. Default: length
        mamode: MA type. Default: 'sma'
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: VHM expression
    """
    from polars_ti.ma import pl_ma
    
    volume_expr = v_expr(volume)
    if volume_expr is None:
        return None
    
    _slength = slength if slength is not None else length
    
    # VHM = (volume - MA(volume)) / rolling_std(volume)
    mu = pl_ma(name=mamode, source=volume_expr, length=length)
    std = volume_expr.rolling_std(window_size=_slength, min_samples=_slength)
    
    vhm_expr = (volume_expr - mu) / std
    
    if offset != 0:
        vhm_expr = vhm_expr.shift(offset)
    
    _name = f"VHM_{length}" if length == _slength else f"VHM_{length}_{_slength}"
    return vhm_expr.alias(_name)

