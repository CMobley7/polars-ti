# -*- coding: utf-8 -*-
from numpy import (
    clip,
    cumsum,
    float64,
    int64,
    isnan,
    nan,
    nan_to_num,
    where,
    zeros_like,
)
from numba import njit

from polars_ti.utils import nb_ffill, nb_idiff


@njit(cache=True)
def nb_exhc(x, n, cap, lb, ub, show_all):
    x_diff = nb_idiff(x, n)
    neg_diff, pos_diff = x_diff < 0, x_diff > 0

    dn_csum = cumsum(neg_diff)
    up_csum = cumsum(pos_diff)

    dn = dn_csum - nb_ffill(where(~neg_diff, dn_csum, nan))
    up = up_csum - nb_ffill(where(~pos_diff, up_csum, nan))

    if cap > 0:
        dn = clip(dn, 0, cap)
        up = clip(up, 0, cap)

    if show_all:
        dn = where(dn == 0, 0, dn)
        up = where(up == 0, 0, up)
    else:
        between_lu = (dn >= lb) & (dn <= ub)
        dn = where(between_lu, dn, 0)
        up = where(between_lu, up, 0)

    return dn, up


# =============================================================================
# Polars EXHC (Exhaustion Count) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def exhc(
    close: IntoExpr,
    length: int = 4,
    cap: int = 13,
    show_all: bool = True,
    asint: bool = False,
    nozeros: bool = False,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Exhaustion Count (EXHC)

    Inspired by Tom DeMark's Sequential - identifies where trends exhaust.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Difference for sequences. Default: 4
        cap: Max sequence number (0 = no cap). Default: 13
        show_all: Show 1-13, if False show 6-9. Default: True
        asint: Convert to int. Default: False
        nozeros: Replace zeros with null. Default: False
        offset: Shift result. Default: 0

    Returns:
        list[pl.Expr]: [EXHC_DN, EXHC_UP] expressions
    """
    close_expr = v_expr(close)
    _length = length
    _cap = cap
    _show_all = show_all

    def compute_exhc(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        dn, up = nb_exhc(arr, _length, _cap, 6, 9, _show_all)
        # Return as struct
        return pl.Series([{"dn": d, "up": u} for d, u in zip(dn, up)])

    struct_expr = close_expr.map_batches(compute_exhc, return_dtype=pl.Struct({"dn": pl.Float64, "up": pl.Float64}))

    dn_expr = struct_expr.struct.field("dn")
    up_expr = struct_expr.struct.field("up")

    if asint:
        dn_expr = dn_expr.cast(pl.Int64)
        up_expr = up_expr.cast(pl.Int64)

    if nozeros:
        dn_expr = pl.when(dn_expr == 0).then(None).otherwise(dn_expr)
        up_expr = pl.when(up_expr == 0).then(None).otherwise(up_expr)

    if offset != 0:
        dn_expr = dn_expr.shift(offset)
        up_expr = up_expr.shift(offset)

    suffix = "a" if show_all else ""
    return [dn_expr.alias(f"EXHC_DN{suffix}"), up_expr.alias(f"EXHC_UP{suffix}")]
