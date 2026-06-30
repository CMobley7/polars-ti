# -*- coding: utf-8 -*-
# =============================================================================
# Polars VWAP (Volume Weighted Average Price) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def vwap(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    datetime_col: IntoExpr | None = None,
    anchor: str = "1d",
    bands: list[float] | None = None,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Volume Weighted Average Price (VWAP)

    VWAP with anchor period support (resets each period) and stddev bands.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        datetime_col: Column name or pl.Expr for datetime. Required for anchoring.
            If None, cumulative VWAP without resets.
        anchor: Polars truncation string for period anchoring.
            Examples: "1d" (daily), "1w" (weekly), "1mo" (monthly), "1h" (hourly).
            Default: "1d"
        bands: List of stddev multipliers for bands. E.g., [1, 2] creates ±1σ and ±2σ.
            Default: None (no bands)
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: [VWAP] or [VWAP, VWAP_L_1, VWAP_U_1, ...] if bands provided
    """
    from polars_ti.overlap.hlc3 import hlc3

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if any(x is None for x in [high_expr, low_expr, close_expr, volume_expr]):
        return None

    # Typical price = HLC3
    tp = hlc3(high_expr, low_expr, close_expr)
    tp_vol = tp * volume_expr

    _anchor = anchor.upper() if anchor else "D"
    _props = f"VWAP_{_anchor}"

    if datetime_col is not None:
        # Anchored VWAP with period resets
        dt_expr = v_expr(datetime_col)
        if dt_expr is None:
            return None

        # Create period group column
        period = dt_expr.dt.truncate(anchor)

        # Cumulative sums within each period
        tp_vol_cum = tp_vol.cum_sum().over(period)
        vol_cum = volume_expr.cum_sum().over(period)

        vwap_expr = tp_vol_cum / vol_cum

        if bands:
            # Variance calculation for stddev bands
            vwap_var = volume_expr * (tp - vwap_expr).pow(2)
            vwap_var_cum = vwap_var.cum_sum().over(period)
            std_weighted = (vwap_var_cum / vol_cum).sqrt()
    else:
        # Cumulative VWAP without resets
        vwap_expr = tp_vol.cum_sum() / volume_expr.cum_sum()

        if bands:
            # Variance calculation for stddev bands (cumulative)
            vwap_var = volume_expr * (tp - vwap_expr).pow(2)
            vwap_var_cum = vwap_var.cum_sum()
            vol_cum = volume_expr.cum_sum()
            std_weighted = (vwap_var_cum / vol_cum).sqrt()

    if offset != 0:
        vwap_expr = vwap_expr.shift(offset)
        if bands:
            std_weighted = std_weighted.shift(offset)

    result = [vwap_expr.alias(_props)]

    if bands:
        for band in bands:
            lower = vwap_expr - band * std_weighted
            upper = vwap_expr + band * std_weighted
            result.append(lower.alias(f"{_props}_L_{band}"))
            result.append(upper.alias(f"{_props}_U_{band}"))

    return result
