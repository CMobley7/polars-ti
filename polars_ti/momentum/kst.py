# -*- coding: utf-8 -*-
# =============================================================================
# Polars KST Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def kst(
    close: IntoExpr,
    signal: int = 9,
    roc1: int = 10,
    roc2: int = 15,
    roc3: int = 20,
    roc4: int = 30,
    sma1: int = 10,
    sma2: int = 10,
    sma3: int = 10,
    sma4: int = 15,
    offset: int = 0,
) -> PlExpr:
    """Polars: Know Sure Thing (KST)

    Momentum oscillator based on ROC at four different periods.

    Formula:
        ROCMA1 = SMA(ROC(close, roc1), sma1)
        KST = 100 * (ROCMA1 + 2*ROCMA2 + 3*ROCMA3 + 4*ROCMA4)
        KST_Signal = SMA(KST, signal)

    Args:
        close: Column name or pl.Expr for input values
        signal: Signal period. Default: 9
        roc1-4: ROC periods. Default: 10, 15, 20, 30
        sma1-4: SMA periods. Default: 10, 10, 10, 15
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with KST and KSTs columns
    """
    from polars_ti.momentum.roc import roc

    close_expr = v_expr(close)

    rocma1 = roc(close_expr, length=roc1).rolling_mean(window_size=sma1)
    rocma2 = roc(close_expr, length=roc2).rolling_mean(window_size=sma2)
    rocma3 = roc(close_expr, length=roc3).rolling_mean(window_size=sma3)
    rocma4 = roc(close_expr, length=roc4).rolling_mean(window_size=sma4)

    kst_expr = 100.0 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    kst_signal_expr = kst_expr.rolling_mean(window_size=signal)

    if offset != 0:
        kst_expr = kst_expr.shift(offset)
        kst_signal_expr = kst_signal_expr.shift(offset)

    _name = f"KST_{roc1}_{roc2}_{roc3}_{roc4}_{sma1}_{sma2}_{sma3}_{sma4}"
    return pl.struct(
        kst_expr.alias(_name),
        kst_signal_expr.alias(f"KSTs_{signal}"),
    ).alias(f"{_name}_{signal}")
