# -*- coding: utf-8 -*-
# =============================================================================
# Polars Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def smi(
    close: IntoExpr = "close",
    fast: int = 5,
    slow: int = 20,
    signal: int = 5,
    scalar: float = 1.0,
    offset: int = 0,
) -> PlExpr:
    """Polars: SMI Ergodic Indicator (SMI)

    The SMI Ergodic Indicator is the same as the True Strength Index (TSI)
    developed by William Blau, except the SMI includes a signal line.
    The SMI uses double moving averages of price minus previous price
    over 2 time frames.

    Sources:
        https://www.motivewave.com/studies/smi_ergodic_indicator.htm
        https://www.tradingview.com/script/Xh5Q0une-SMI-Ergodic-Oscillator/

    Args:
        close (IntoExpr): Column name or expression for 'close'. Default: "close"
        fast (int): The short period. Default: 5
        slow (int): The long period. Default: 20
        signal (int): The signal period. Default: 5
        scalar (float): How much to magnify. Default: 1
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pl.Expr: Struct expression with columns: SMI, SMIs (signal), SMIo (oscillator)
    """
    from polars_ti.momentum.tsi import tsi

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    if slow < fast:
        fast, slow = slow, fast

    _props = f"_{fast}_{slow}_{signal}_{scalar}"

    # Get TSI expressions (SMI is TSI with different defaults + oscillator)
    tsi_exprs = tsi(
        close=close_expr,
        fast=fast,
        slow=slow,
        signal=signal,
        scalar=scalar,
        mamode="ema",
        drift=1,
        offset=0,  # We'll apply offset at the end
    )

    if tsi_exprs is None:
        return None

    # TSI returns [tsi, tsi_signal]
    smi_expr = tsi_exprs[0]
    signalma_expr = tsi_exprs[1]

    # Calculate oscillator = SMI - Signal
    osc_expr = smi_expr - signalma_expr

    # Apply offset
    if offset != 0:
        smi_expr = smi_expr.shift(offset)
        signalma_expr = signalma_expr.shift(offset)
        osc_expr = osc_expr.shift(offset)

    # Return as struct
    return pl.struct(
        smi_expr.alias(f"SMI{_props}"),
        signalma_expr.alias(f"SMIs{_props}"),
        osc_expr.alias(f"SMIo{_props}"),
    ).alias(f"SMI{_props}")
