# -*- coding: utf-8 -*-
# =============================================================================
# Polars IFISHER Implementation (Pure Native Polars)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_ifisher(
    close: IntoExpr,
    amp: float = 1.0,
    signal_offset: int = -1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Inverse Fisher Transform

    Changes the Probability Distribution Function for normalized oscillators
    to receive clearer signals. Input should be in range -1 to 1.

    Uses pure native Polars expressions.

    Formula: y = (exp(amp*x) - 1) / (exp(amp*x) + 1)

    Sources:
        https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf
        Book: Cycle Analytics for Traders, John Ehlers (2014)

    Args:
        close: Column name or pl.Expr for input (ideally -1 to 1 range)
        amp: Amplifying factor. Default: 1.0
        signal_offset: Offset for signal line. Default: -1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Inverse Fisher Transform expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Pure native Polars: Inverse Fisher Transform
    # y = (exp(amp*x) - 1) / (exp(amp*x) + 1)
    amped = (close_expr * amp).exp()
    result = (amped - 1) / (amped + 1)

    # Apply offsets
    if offset != 0:
        result = result.shift(offset)
    if signal_offset != 0:
        result = result.shift(signal_offset)

    return result.alias(f"INVFISHER_{amp}")
