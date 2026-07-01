# -*- coding: utf-8 -*-
# =============================================================================
# Polars HT_SINE Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def ht_sine(
    close: IntoExpr,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Hilbert Transform Sine Wave (HT_SINE)

    Uses the Hilbert Transform to compute a sine wave and a leading sine
    wave from the dominant cycle phase, useful for cycle timing signals.

    Sources:
        John Ehlers, "Rocket Science for Traders" (2002)
        TA-Lib: HT_SINE

    Args:
        close: Column name or pl.Expr for 'close' prices.
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with fields:
            - sine:     sin(dcphase) (Float64)
            - leadsine: sin(dcphase + 45°) (Float64)
    """
    close_expr = v_expr(close)
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    from polars_ti.cycles._ht_pipeline import nb_ht_pipeline

    _use_talib = Imports["talib"] and v_talib(talib)

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        if _use_talib:
            from talib import HT_SINE

            sine_arr, leadsine_arr = HT_SINE(arr)
        else:
            _, dcphase, _, _, _ = nb_ht_pipeline(arr)
            dcphase = dcphase.copy()
            dcphase[:63] = np.nan
            sine_arr = np.sin(dcphase * np.pi / 180.0)
            leadsine_arr = np.sin((dcphase + 45.0) * np.pi / 180.0)

        return pl.DataFrame(
            {
                "sine": sine_arr,
                "leadsine": leadsine_arr,
            }
        ).to_struct("HT_SINE")

    result_expr = close_expr.map_batches(
        _compute,
        return_dtype=pl.Struct(
            [
                pl.Field("sine", pl.Float64),
                pl.Field("leadsine", pl.Float64),
            ]
        ),
    )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("HT_SINE")
