# -*- coding: utf-8 -*-
# =============================================================================
# Polars Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def squeeze_pro(
    high: IntoExpr = "high",
    low: IntoExpr = "low",
    close: IntoExpr = "close",
    bb_length: int = 20,
    bb_std: float = 2.0,
    kc_length: int = 20,
    kc_scalar_wide: float = 2.0,
    kc_scalar_normal: float = 1.5,
    kc_scalar_narrow: float = 1.0,
    mom_length: int = 12,
    mom_smooth: int = 6,
    mamode: str = "sma",
    use_tr: bool = True,
    asint: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Squeeze PRO (SQZPRO)

    Extended version of "TTM Squeeze" from John Carter with multiple
    Keltner Channel scalars (wide, normal, narrow).

    Sources:
        https://usethinkscript.com/threads/john-carters-squeeze-pro-indicator-for-thinkorswim-free.4021/
        https://www.tradingview.com/script/TAAt6eRX-Squeeze-PRO-Indicator-Makit0/

    Args:
        high (IntoExpr): Column name or expression for 'high'. Default: "high"
        low (IntoExpr): Column name or expression for 'low'. Default: "low"
        close (IntoExpr): Column name or expression for 'close'. Default: "close"
        bb_length (int): Bollinger Bands period. Default: 20
        bb_std (float): Bollinger Bands Std. Dev. Default: 2
        kc_length (int): Keltner Channel period. Default: 20
        kc_scalar_wide (float): Keltner Channel scalar for wider channel. Default: 2
        kc_scalar_normal (float): Keltner Channel scalar for normal channel. Default: 1.5
        kc_scalar_narrow (float): Keltner Channel scalar for narrow channel. Default: 1
        mom_length (int): Momentum Period. Default: 12
        mom_smooth (int): Smoothing Period of Momentum. Default: 6
        mamode (str): Only "ema" or "sma". Default: "sma"
        use_tr (bool): Use True Range for Keltner Channels. Default: True
        asint (bool): Use integers instead of bool. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pl.Expr: Struct with SQZPRO, SQZPRO_ON_WIDE, SQZPRO_ON_NORMAL,
            SQZPRO_ON_NARROW, SQZPRO_OFF, SQZPRO_NO columns
    """
    from polars_ti.volatility.bbands import bbands
    from polars_ti.volatility.kc import kc
    from polars_ti.momentum.mom import mom
    from polars_ti.ma import ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    # Validate kc scalars
    if not (kc_scalar_wide > kc_scalar_normal > kc_scalar_narrow):
        return None

    _props = "" if use_tr else "hlr"
    _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar_wide}_{kc_scalar_normal}_{kc_scalar_narrow}"

    # Calculate Bollinger Bands
    bb_struct = bbands(close_expr, length=bb_length, std=bb_std, talib=False, offset=0)
    bb_lower_name = f"BBL_{bb_length}_{bb_std}"
    bb_upper_name = f"BBU_{bb_length}_{bb_std}"

    # Calculate three Keltner Channels
    kc_wide = kc(
        high_expr,
        low_expr,
        close_expr,
        length=kc_length,
        scalar=kc_scalar_wide,
        mamode=mamode,
        talib=True,
        tr=use_tr,
        offset=0,
    )
    kc_normal = kc(
        high_expr,
        low_expr,
        close_expr,
        length=kc_length,
        scalar=kc_scalar_normal,
        mamode=mamode,
        talib=True,
        tr=use_tr,
        offset=0,
    )
    kc_narrow = kc(
        high_expr,
        low_expr,
        close_expr,
        length=kc_length,
        scalar=kc_scalar_narrow,
        mamode=mamode,
        talib=True,
        tr=use_tr,
        offset=0,
    )

    # Calculate momentum component
    momo = mom(close_expr, length=mom_length, talib=False, offset=0)
    sqz_val = ma(name=mamode, source=momo, length=mom_smooth, talib=False)

    def compute_squeeze_pro(df_struct: pl.DataFrame) -> pl.Series:
        """Compute squeeze pro using struct fields."""
        bb_l = df_struct["bb_l"].to_numpy()
        bb_u = df_struct["bb_u"].to_numpy()

        kc_wide_l = df_struct["kc_wide_l"].to_numpy()
        kc_wide_u = df_struct["kc_wide_u"].to_numpy()
        kc_normal_l = df_struct["kc_normal_l"].to_numpy()
        kc_normal_u = df_struct["kc_normal_u"].to_numpy()
        kc_narrow_l = df_struct["kc_narrow_l"].to_numpy()
        kc_narrow_u = df_struct["kc_narrow_u"].to_numpy()

        sqz = df_struct["sqz_val"].to_numpy()

        # Classify squeezes
        squeeze_on_wide = (bb_l > kc_wide_l) & (bb_u < kc_wide_u)
        squeeze_on_normal = (bb_l > kc_normal_l) & (bb_u < kc_normal_u)
        squeeze_on_narrow = (bb_l > kc_narrow_l) & (bb_u < kc_narrow_u)
        squeeze_off_wide = (bb_l < kc_wide_l) & (bb_u > kc_wide_u)
        no_squeeze = ~squeeze_on_wide & ~squeeze_off_wide

        if asint:
            squeeze_on_wide = squeeze_on_wide.astype(np.int64)
            squeeze_on_normal = squeeze_on_normal.astype(np.int64)
            squeeze_on_narrow = squeeze_on_narrow.astype(np.int64)
            squeeze_off_wide = squeeze_off_wide.astype(np.int64)
            no_squeeze = no_squeeze.astype(np.int64)

        if offset != 0:
            sqz = np.roll(sqz, offset)
            squeeze_on_wide = np.roll(squeeze_on_wide, offset)
            squeeze_on_normal = np.roll(squeeze_on_normal, offset)
            squeeze_on_narrow = np.roll(squeeze_on_narrow, offset)
            squeeze_off_wide = np.roll(squeeze_off_wide, offset)
            no_squeeze = np.roll(no_squeeze, offset)
            if offset > 0:
                sqz[:offset] = np.nan
                if asint:
                    squeeze_on_wide[:offset] = 0
                    squeeze_on_normal[:offset] = 0
                    squeeze_on_narrow[:offset] = 0
                    squeeze_off_wide[:offset] = 0
                    no_squeeze[:offset] = 0
            else:
                sqz[offset:] = np.nan
                if asint:
                    squeeze_on_wide[offset:] = 0
                    squeeze_on_normal[offset:] = 0
                    squeeze_on_narrow[offset:] = 0
                    squeeze_off_wide[offset:] = 0
                    no_squeeze[offset:] = 0

        return pl.DataFrame(
            {
                f"SQZPRO{_props}": sqz,
                "SQZPRO_ON_WIDE": squeeze_on_wide,
                "SQZPRO_ON_NORMAL": squeeze_on_normal,
                "SQZPRO_ON_NARROW": squeeze_on_narrow,
                "SQZPRO_OFF": squeeze_off_wide,
                "SQZPRO_NO": no_squeeze,
            }
        ).to_struct(f"SQZPRO{_props}")

    on_dtype = pl.Int64 if asint else pl.Boolean
    return_dtype = pl.Struct(
        [
            pl.Field(f"SQZPRO{_props}", pl.Float64),
            pl.Field("SQZPRO_ON_WIDE", on_dtype),
            pl.Field("SQZPRO_ON_NORMAL", on_dtype),
            pl.Field("SQZPRO_ON_NARROW", on_dtype),
            pl.Field("SQZPRO_OFF", on_dtype),
            pl.Field("SQZPRO_NO", on_dtype),
        ]
    )

    combined = pl.struct(
        [
            bb_struct.struct.field(bb_lower_name).alias("bb_l"),
            bb_struct.struct.field(bb_upper_name).alias("bb_u"),
            kc_wide.struct.field("kcl").alias("kc_wide_l"),
            kc_wide.struct.field("kcu").alias("kc_wide_u"),
            kc_normal.struct.field("kcl").alias("kc_normal_l"),
            kc_normal.struct.field("kcu").alias("kc_normal_u"),
            kc_narrow.struct.field("kcl").alias("kc_narrow_l"),
            kc_narrow.struct.field("kcu").alias("kc_narrow_u"),
            sqz_val.alias("sqz_val"),
        ]
    )

    return combined.map_batches(lambda s: compute_squeeze_pro(s.struct.unnest()), return_dtype=return_dtype).alias(
        f"SQZPRO{_props}"
    )
