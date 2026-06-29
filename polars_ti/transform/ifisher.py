# -*- coding: utf-8 -*-
# =============================================================================
# Polars IFISHER Implementation (Pure Native Polars)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def ifisher(
    close: IntoExpr,
    amp: float = 1.0,
    signal_offset: int = -1,
    offset: int = 0,
) -> list[pl.Expr]:
    """Polars: Inverse Fisher Transform

    Changes the Probability Distribution Function for normalized oscillators
    to receive clearer signals. The transform requires its input to lie in the
    range [-1, 1]; when any value falls outside that range the whole series is
    first linearly remapped to [-1, 1] using its fixed full-series min/max
    (matching pandas-ta's ``ifisher``/``remap``), otherwise raw prices would
    saturate ``exp(amp*x)`` to ≈1 for every bar.

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
        list[pl.Expr]: ``[INVFISHER_{amp}, INVFISHERs_{amp}]`` expressions.
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Remap the input to [-1, 1] when any value falls outside it, using the
    # fixed full-series min/max (pandas-ta remap: -1 + 2*(x-min)/(max-min)).
    mn = close_expr.min()
    mx = close_expr.max()
    all_in_range = close_expr.is_between(-1, 1).all()
    remapped = -1.0 + (2.0 / (mx - mn)) * (close_expr - mn)
    x = pl.when(all_in_range).then(close_expr).otherwise(remapped)

    # Inverse Fisher Transform: y = (exp(amp*x) - 1) / (exp(amp*x) + 1)
    amped = (x * amp).exp()
    result = (amped - 1) / (amped + 1)

    # OLD applies BOTH offset and signal_offset to BOTH lines, so the main and
    # signal series are identical (preserved for parity).
    inv_fisher = result
    signal = result
    if offset != 0:
        inv_fisher = inv_fisher.shift(offset)
        signal = signal.shift(offset)
    if signal_offset != 0:
        inv_fisher = inv_fisher.shift(signal_offset)
        signal = signal.shift(signal_offset)

    return [
        inv_fisher.alias(f"INVFISHER_{amp}"),
        signal.alias(f"INVFISHERs_{amp}"),
    ]
