# -*- coding: utf-8 -*-
# =============================================================================
# Polars ICHIMOKU Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.midprice import pl_midprice


def pl_ichimoku(
    df: pl.DataFrame,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    include_chikou: bool = True,
    lookahead: bool = True,
    offset: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Polars: Ichimoku Kinkō Hyō

    Developed Pre WWII as a forecasting model for financial markets.

    WARNING: This function may leak future data when used for machine learning
        if lookahead=True (default). Set lookahead=False to avoid data leakage.

    Args:
        df: Polars DataFrame with price columns
        high: Column name for 'high' prices. Default: "high"
        low: Column name for 'low' prices. Default: "low"
        close: Column name for 'close' prices. Default: "close"
        tenkan: Tenkan period. Default: 9
        kijun: Kijun period. Default: 26
        senkou: Senkou period. Default: 52
        include_chikou: Whether to include chikou span. Default: True
        lookahead: If False, excludes chikou span to prevent data leakage. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]:
            - Main DataFrame with ISA, ISB, ITS, IKS, and optionally ICS columns
            - Span DataFrame with future ISA and ISB values
    """
    # Handle lookahead parameter
    if not lookahead:
        include_chikou = False

    # Calculate the components using pl_midprice
    tenkan_sen = df.select(pl_midprice(high, low, length=tenkan, talib=False)).get_column(f"MIDPRICE_{tenkan}")
    kijun_sen = df.select(pl_midprice(high, low, length=kijun, talib=False)).get_column(f"MIDPRICE_{kijun}")
    senkou_b = df.select(pl_midprice(high, low, length=senkou, talib=False)).get_column(f"MIDPRICE_{senkou}")

    # Span A = (tenkan_sen + kijun_sen) / 2
    span_a = (tenkan_sen + kijun_sen) / 2

    # Copy values before shift for future span
    span_a_future = span_a[-kijun:].shift(-1)
    span_b_future = senkou_b[-kijun:].shift(-1)

    # Shift spans forward by kijun - 1
    span_a_shifted = span_a.shift(kijun - 1)
    span_b_shifted = senkou_b.shift(kijun - 1)

    # Chikou span = close shifted backward
    close_col = df.get_column(close)
    chikou_span = close_col.shift(-kijun + 1)

    # Apply offset if needed
    if offset != 0:
        tenkan_sen = tenkan_sen.shift(offset)
        kijun_sen = kijun_sen.shift(offset)
        span_a_shifted = span_a_shifted.shift(offset)
        span_b_shifted = span_b_shifted.shift(offset)
        chikou_span = chikou_span.shift(offset)

    # Build main DataFrame
    data = {
        f"ISA_{tenkan}": span_a_shifted,
        f"ISB_{kijun}": span_b_shifted,
        f"ITS_{tenkan}": tenkan_sen,
        f"IKS_{kijun}": kijun_sen,
    }
    if include_chikou:
        data[f"ICS_{kijun}"] = chikou_span

    ichimoku_df = pl.DataFrame(data)

    # Build future span DataFrame
    span_df = pl.DataFrame(
        {
            f"ISA_{tenkan}": span_a_future.to_list(),
            f"ISB_{kijun}": span_b_future.to_list(),
        }
    )

    return ichimoku_df, span_df
