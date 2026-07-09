# -*- coding: utf-8 -*-
# =============================================================================
# Polars ICHIMOKU Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.midprice import midprice


def ichimoku(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    include_chikou: bool = True,
    lookahead: bool = True,
    offset: int = 0,
    forward: bool = False,
) -> pl.Expr:
    """Polars: Ichimoku Kinkō Hyō

    Developed Pre WWII as a forecasting model for financial markets.

    WARNING: This function may leak future data when used for machine learning
        if lookahead=True (default). Set lookahead=False to avoid data leakage.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        tenkan: Tenkan period. Default: 9
        kijun: Kijun period. Default: 26
        senkou: Senkou period. Default: 52
        include_chikou: Whether to include chikou span. Default: True
        lookahead: If False, excludes chikou span to prevent data leakage. Default: True
        offset: Shift result by N periods. Default: 0
        forward: If True, also emit the forward-projected ("future cloud") Senkou
            Span A/B as ``ISA_<tenkan>_F`` / ``ISB_<kijun>_F``. These carry the
            un-shifted span values that OLD pandas-ta returned as its separate
            forward DataFrame; the future cloud is obtained by projecting them
            ``kijun`` bars ahead (e.g. ``col.tail(kijun).shift(-1)``). Since a
            Polars expression cannot append future rows, the projection is
            exposed as row-aligned columns instead. Default: False

    Returns:
        pl.Expr: Struct expression with ISA, ISB, ITS, IKS, optionally ICS, and
            optionally the forward-projection columns when ``forward=True``.
    """
    if not lookahead:
        include_chikou = False

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    # Tenkan-sen, Kijun-sen, Senkou Span B from midprice (native, like old)
    tenkan_sen = midprice(high_expr, low_expr, length=tenkan, talib=False)
    kijun_sen = midprice(high_expr, low_expr, length=kijun, talib=False)
    senkou_b = midprice(high_expr, low_expr, length=senkou, talib=False)

    # Span A = (tenkan + kijun) / 2
    span_a = 0.5 * (tenkan_sen + kijun_sen)

    # Spans shifted forward by kijun - 1 (Polars shift moves later rows down)
    span_a_shifted = span_a.shift(kijun - 1)
    span_b_shifted = senkou_b.shift(kijun - 1)

    # Chikou span = close shifted backward
    chikou_span = close_expr.shift(-kijun + 1)

    # Forward-projection ("future cloud") spans: the un-shifted Span A/B that
    # OLD pandas-ta returned in its separate forward DataFrame.
    span_a_forward = span_a
    span_b_forward = senkou_b

    if offset != 0:
        tenkan_sen = tenkan_sen.shift(offset)
        kijun_sen = kijun_sen.shift(offset)
        span_a_shifted = span_a_shifted.shift(offset)
        span_b_shifted = span_b_shifted.shift(offset)
        chikou_span = chikou_span.shift(offset)
        span_a_forward = span_a_forward.shift(offset)
        span_b_forward = span_b_forward.shift(offset)

    isa_name = f"ISA_{tenkan}"
    isb_name = f"ISB_{kijun}"
    its_name = f"ITS_{tenkan}"
    iks_name = f"IKS_{kijun}"
    ics_name = f"ICS_{kijun}"

    struct_fields = [
        span_a_shifted.alias(isa_name),
        span_b_shifted.alias(isb_name),
        tenkan_sen.alias(its_name),
        kijun_sen.alias(iks_name),
    ]
    if include_chikou:
        struct_fields.append(chikou_span.alias(ics_name))
    if forward:
        struct_fields.append(span_a_forward.alias(f"{isa_name}_F"))
        struct_fields.append(span_b_forward.alias(f"{isb_name}_F"))

    return pl.struct(struct_fields).alias(f"ICHIMOKU_{tenkan}_{kijun}_{senkou}")
