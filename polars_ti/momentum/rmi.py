# -*- coding: utf-8 -*-
# =============================================================================
# Polars RMI (Relative Momentum Index) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.ma import ma


def rmi(
    close: IntoExpr,
    length: int = 14,
    momentum: int = 5,
    scalar: float = 100.0,
    mamode: str = "rma",
    offset: int = 0,
) -> PlExpr:
    """Polars: Relative Momentum Index (RMI)

    RMI is a momentum oscillator similar to RSI, but measures momentum
    over multiple periods instead of single-period price changes.

    Sources:
        https://www.investopedia.com/terms/r/relative_momentum_index.asp

    Calculation:
        momentum_change = close.diff(momentum)
        gain = momentum_change.clip(lower=0)
        loss = (-momentum_change).clip(lower=0)
        avg_gain = ma(mamode, gain, length)
        avg_loss = ma(mamode, loss, length)
        RMI = 100 - (100 / (1 + avg_gain / avg_loss))

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Smoothing period. Default: 14
        momentum: Lookback period for momentum change. Default: 5
        scalar: Multiplication factor. Default: 100
        mamode: MA type for smoothing. Default: 'rma' (Wilder's)
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: RMI expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Momentum-based price changes (positive = gain, negative = loss)
    diff_expr = close_expr.diff(momentum)
    gain_expr = diff_expr.clip(lower_bound=0.0)
    loss_expr = (-diff_expr).clip(lower_bound=0.0)

    # Smooth gains and losses using the specified MA
    avg_gain_expr = ma(mamode, gain_expr, length=length)
    avg_loss_expr = ma(mamode, loss_expr, length=length)

    # RMI = scalar - (scalar / (1 + avg_gain / avg_loss))
    rs_expr = avg_gain_expr / avg_loss_expr
    rmi_expr = scalar - (scalar / (1 + rs_expr))

    if offset != 0:
        rmi_expr = rmi_expr.shift(offset)

    return rmi_expr.alias(f"RMI_{length}_{momentum}")
