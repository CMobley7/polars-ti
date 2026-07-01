# -*- coding: utf-8 -*-
# =============================================================================
# Polars HT_PHASOR Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def ht_phasor(
    close: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Hilbert Transform Phasor Components (HT_PHASOR)

    Uses the Hilbert Transform to return the in-phase and quadrature
    (phasor) components of the dominant cycle.

    Sources:
        John Ehlers, "Rocket Science for Traders" (2002)
        TA-Lib: HT_PHASOR

    Args:
        close: Column name or pl.Expr for 'close' prices.
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with fields:
            - inphase:    In-phase component (Float64)
            - quadrature: Quadrature component (Float64)
    """
    close_expr = v_expr(close)
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    from polars_ti.cycles._ht_pipeline import nb_ht_pipeline

    _use_talib = Imports["talib"] and v_talib(talib)

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        if _use_talib:
            from talib import HT_PHASOR

            inphase_arr, quad_arr = HT_PHASOR(arr)
        else:
            _, _, inphase_arr, quad_arr, _ = nb_ht_pipeline(arr)
            inphase_arr = inphase_arr.copy()
            quad_arr = quad_arr.copy()
            inphase_arr[:32] = np.nan
            quad_arr[:32] = np.nan

        return pl.DataFrame(
            {
                "inphase": inphase_arr,
                "quadrature": quad_arr,
            }
        ).to_struct("HT_PHASOR")

    result_expr = close_expr.map_batches(
        _compute,
        return_dtype=pl.Struct(
            [
                pl.Field("inphase", pl.Float64),
                pl.Field("quadrature", pl.Float64),
            ]
        ),
    )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("HT_PHASOR")
