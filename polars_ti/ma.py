# -*- coding: utf-8 -*-
# =============================================================================
# Polars MA Dispatcher
# =============================================================================
from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.overlap.dema import dema
from polars_ti.overlap.ema import ema
from polars_ti.overlap.fwma import fwma
from polars_ti.overlap.hma import hma
from polars_ti.overlap.linreg import linreg
from polars_ti.overlap.midpoint import midpoint
from polars_ti.overlap.pwma import pwma
from polars_ti.overlap.rma import rma
from polars_ti.overlap.sinwma import sinwma
from polars_ti.overlap.sma import sma
from polars_ti.overlap.ssf import ssf
from polars_ti.overlap.swma import swma
from polars_ti.overlap.t3 import t3
from polars_ti.overlap.tema import tema
from polars_ti.overlap.trima import trima
from polars_ti.overlap.vidya import vidya
from polars_ti.overlap.wma import wma


# MA function mapping
_PL_MA_FUNCS = {
    "dema": dema,
    "ema": ema,
    "fwma": fwma,
    "hma": hma,
    "linreg": linreg,
    "midpoint": midpoint,
    "pwma": pwma,
    "rma": rma,
    "sinwma": sinwma,
    "sma": sma,
    "ssf": ssf,
    "swma": swma,
    "t3": t3,
    "tema": tema,
    "trima": trima,
    "vidya": vidya,
    "wma": wma,
}


def ma(
    name: str = "ema",
    source: IntoExpr = None,
    length: int = 10,
    talib: bool = True,
    **kwargs,
) -> PlExpr:
    """Polars: Simple MA Utility for easier MA selection

    Available MAs:
        dema, ema, fwma, hma, linreg, midpoint, pwma, rma, sinwma, sma, ssf,
        swma, t3, tema, trima, vidya, wma

    Examples:
        ema8 = pl_ma("ema", "close", length=8)
        sma50 = pl_ma("sma", "close", length=50)
        pwma10 = pl_ma("pwma", "close", length=10, asc=False)

    Args:
        name: One of the Available MAs. Default: "ema"
        source: Column name or pl.Expr for the source data
        length: Rolling window period. Default: 10
        talib: If True and TA-Lib is installed, uses TA-Lib. Default: True

    Kwargs:
        Any additional kwargs the MA may require (offset, asc, etc.)

    Returns:
        pl.Expr: MA expression for lazy evaluation
    """
    if name is None and source is None:
        return list(_PL_MA_FUNCS.keys())

    name = name.lower() if isinstance(name, str) else "ema"

    if name not in _PL_MA_FUNCS:
        name = "ema"  # Default fallback

    ma_func = _PL_MA_FUNCS[name]

    # Build kwargs based on what the function accepts
    # All MAs accept: close/source, length, offset
    # Some accept: talib, asc, presma, etc.
    call_kwargs = {"length": length}

    # Add talib param for MAs that support it
    if name in (
        "sma",
        "ema",
        "wma",
        "dema",
        "tema",
        "t3",
        "trima",
        "linreg",
        "midpoint",
        "vidya",
    ):
        call_kwargs["talib"] = talib

    # Add presma param only for MAs that support it (ema, rma, t3, tema)
    # T3 and TEMA forward presma to their internal EMA calls
    # Always pop presma from kwargs to prevent passing it to MAs that don't support it
    presma_val = kwargs.pop("presma", None)
    if name in ("ema", "rma", "t3", "tema") and presma_val is not None:
        call_kwargs["presma"] = presma_val

    # Add remaining kwargs
    call_kwargs.update(kwargs)

    return ma_func(source, **call_kwargs)
