# -*- coding: utf-8 -*-
# =============================================================================
# Polars ALLIGATOR Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.smma import smma


def alligator(
    close: IntoExpr,
    jaw: int = 13,
    teeth: int = 8,
    lips: int = 5,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Bill Williams Alligator

    Three SMMA lines representing Jaw, Teeth, and Lips.
    Returns a struct with AGj, AGt, AGl fields.

    Args:
        close: Column name or pl.Expr for 'close'
        jaw: Jaw period. Default: 13
        teeth: Teeth period. Default: 8
        lips: Lips period. Default: 5
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with AGj, AGt, AGl fields
    """
    close_expr = v_expr(close)
    _props = f"_{jaw}_{teeth}_{lips}"

    # Build three SMMA expressions
    jaw_expr = smma(close, length=jaw, offset=offset).alias(f"AGj{_props}")
    teeth_expr = smma(close, length=teeth, offset=offset).alias(f"AGt{_props}")
    lips_expr = smma(close, length=lips, offset=offset).alias(f"AGl{_props}")

    return pl.struct([jaw_expr, teeth_expr, lips_expr]).alias(f"AG{_props}")
