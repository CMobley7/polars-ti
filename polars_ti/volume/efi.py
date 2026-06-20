# -*- coding: utf-8 -*-
# =============================================================================
# Polars EFI (Elder's Force Index) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def efi(
    close: IntoExpr,
    volume: IntoExpr,
    length: int = 13,
    mamode: str = "ema",
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Elder's Force Index (EFI)

    Elder's Force Index measures the power behind a price movement using
    price and volume as well as potential reversals and price corrections.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        length: MA smoothing period. Default: 13
        mamode: MA type for smoothing. Default: 'ema'
        drift: The diff period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: EFI expression
    """
    from polars_ti.ma import ma

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    # EFI = MA(close.diff(drift) * volume, length)
    pv_diff = close_expr.diff(drift) * volume_expr
    efi_expr = ma(name=mamode, source=pv_diff, length=length)

    if offset != 0:
        efi_expr = efi_expr.shift(offset)

    return efi_expr.alias(f"EFI_{length}")
