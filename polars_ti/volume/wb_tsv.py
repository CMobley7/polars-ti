# -*- coding: utf-8 -*-
# =============================================================================
# Polars WB_TSV (Time Segmented Value) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def wb_tsv(
    close: IntoExpr,
    volume: IntoExpr,
    length: int = 18,
    signal: int = 10,
    mamode: str = "sma",
    drift: int = 1,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Time Segmented Value (TSV)

    Worden Brothers proprietary oscillator comparing price and volume
    over time segments.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        length: TSV period. Default: 18
        signal: Signal MA period. Default: 10
        mamode: MA type. Default: 'sma'
        drift: Difference period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: [TSV, TSV_signal, TSV_ratio]
    """
    from polars_ti.ma import ma

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    _props = f"_{length}_{signal}"

    # Signed volume based on close direction
    close_diff = close_expr.diff(1)
    sign = pl.when(close_diff > 0).then(1).when(close_diff < 0).then(-1).otherwise(0)
    signed_volume = volume_expr * sign.abs()  # Use absolute value for signed

    # CVD = signed_volume * diff(close, drift)
    cvd = signed_volume * close_expr.diff(drift)

    # TSV = rolling_sum(cvd, length)
    tsv_expr = cvd.rolling_sum(window_size=length, min_samples=length)

    # Signal = MA(TSV, signal)
    signal_expr = ma(name=mamode, source=tsv_expr, length=signal)

    # Ratio = TSV / Signal (with div/0 protection)
    ratio_expr = tsv_expr / signal_expr

    if offset != 0:
        tsv_expr = tsv_expr.shift(offset)
        signal_expr = signal_expr.shift(offset)
        ratio_expr = ratio_expr.shift(offset)

    return [
        tsv_expr.alias(f"TSV{_props}"),
        signal_expr.alias(f"TSVs{_props}"),
        ratio_expr.alias(f"TSVr{_props}"),
    ]
