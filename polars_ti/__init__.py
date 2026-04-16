"""
.. moduleauthor:: Christopher Mobley
"""

name = "polars_ti"

from polars_ti.candles import CDL_PATTERN_NAMES
from polars_ti.candles import __all__ as candles_all
from polars_ti.candles import pl_cdl, pl_cdl_doji, pl_cdl_inside, pl_cdl_pattern, pl_cdl_z, pl_ha

# Enable "ti" DataFrame Extension
from polars_ti.core import TechnicalIndicators

# Custom External Directory Commands. See help(import_dir)
from polars_ti.custom import create_dir, import_dir
from polars_ti.cycles import __all__ as cycles_all
from polars_ti.cycles import pl_dsp, pl_ebsw, pl_reflex

# Common Averages useful for Indicators
# with a mamode argument, like ti.adx()
from polars_ti.ma import pl_ma

# Dictionaries and version
from polars_ti.maps import EXCHANGE_TZ, RATE, Category, Imports, version
from polars_ti.momentum import __all__ as momentum_all
from polars_ti.momentum import (
    pl_ao,
    pl_apo,
    pl_bias,
    pl_bop,
    pl_brar,
    pl_cci,
    pl_cfo,
    pl_cg,
    pl_cmo,
    pl_coppock,
    pl_crsi,
    pl_cti,
    pl_dm,
    pl_er,
    pl_eri,
    pl_exhc,
    pl_fisher,
    pl_imi,
    pl_inertia,
    pl_kdj,
    pl_kst,
    pl_lrsi,
    pl_macd,
    pl_mom,
    pl_pgo,
    pl_po,
    pl_ppo,
    pl_psl,
    pl_qqe,
    pl_rmi,
    pl_roc,
    pl_rsi,
    pl_rsx,
    pl_rvgi,
    pl_slope,
    pl_smc,
    pl_smi,
    pl_squeeze,
    pl_squeeze_pro,
    pl_stc,
    pl_stoch,
    pl_stochf,
    pl_stochrsi,
    pl_tmo,
    pl_trix,
    pl_trixh,
    pl_tsi,
    pl_uo,
    pl_vwmacd,
    pl_willr,
)
from polars_ti.overlap import __all__ as overlap_all
from polars_ti.overlap import (
    pl_alligator,
    pl_alma,
    pl_dema,
    pl_ema,
    pl_fwma,
    pl_hilo,
    pl_hl2,
    pl_hlc3,
    pl_hma,
    pl_hwma,
    pl_ichimoku,
    pl_jma,
    pl_kama,
    pl_linreg,
    pl_mama,
    pl_mcgd,
    pl_midpoint,
    pl_midprice,
    pl_mmar,
    pl_ohlc4,
    pl_ott,
    pl_pivots,
    pl_pwma,
    pl_rainbow,
    pl_rma,
    pl_sinwma,
    pl_sma,
    pl_smma,
    pl_ssf,
    pl_ssf3,
    pl_supertrend,
    pl_swma,
    pl_t3,
    pl_tema,
    pl_trima,
    pl_vidya,
    pl_wcp,
    pl_wma,
    pl_zlma,
)
from polars_ti.performance import __all__ as performance_all
from polars_ti.performance import pl_drawdown, pl_log_return, pl_percent_return
from polars_ti.statistics import __all__ as statistics_all
from polars_ti.statistics import (
    pl_entropy,
    pl_kurtosis,
    pl_mad,
    pl_median,
    pl_quantile,
    pl_skew,
    pl_stdev,
    pl_tos_stdevall,
    pl_variance,
    pl_zscore,
)
from polars_ti.transform import __all__ as transform_all
from polars_ti.transform import pl_cube, pl_ifisher, pl_remap
from polars_ti.trend import __all__ as trend_all
from polars_ti.trend import (
    pl_adx,
    pl_alphatrend,
    pl_amat,
    pl_aroon,
    pl_chop,
    pl_cksp,
    pl_decay,
    pl_decreasing,
    pl_dpo,
    pl_ht_trendline,
    pl_increasing,
    pl_long_run,
    pl_pmax,
    pl_psar,
    pl_qstick,
    pl_rwi,
    pl_short_run,
    pl_trama,
    pl_trendflex,
    pl_tsignals,
    pl_ttm_trend,
    pl_vhf,
    pl_vortex,
    pl_xsignals,
    pl_zigzag,
)
from polars_ti.volatility import __all__ as volatility_all
from polars_ti.volatility import (
    pl_aberration,
    pl_accbands,
    pl_atr,
    pl_atrts,
    pl_avsl,
    pl_bbands,
    pl_chandelier_exit,
    pl_donchian,
    pl_fvg,
    pl_halftrend,
    pl_hwc,
    pl_kc,
    pl_massi,
    pl_natr,
    pl_pdist,
    pl_rvi,
    pl_thermo,
    pl_true_range,
    pl_ui,
)
from polars_ti.volume import __all__ as volume_all
from polars_ti.volume import (
    pl_ad,
    pl_adosc,
    pl_aobv,
    pl_avwap,
    pl_cmf,
    pl_efi,
    pl_eom,
    pl_kvo,
    pl_mfi,
    pl_nvi,
    pl_obv,
    pl_pvi,
    pl_pvo,
    pl_pvol,
    pl_pvr,
    pl_pvt,
    pl_vfi,
    pl_vhm,
    pl_vp,
    pl_vwap,
    pl_vwma,
    pl_wb_tsv,
)

__all__ = [
    "name",
    "EXCHANGE_TZ",
    "RATE",
    "Category",
    "Imports",
    "version",
    "pl_ma",
    "CDL_PATTERN_NAMES",
    "create_dir",
    "import_dir",
    "TechnicalIndicators",
]

__all__ += (
    candles_all
    + cycles_all
    + momentum_all
    + overlap_all
    + performance_all
    + statistics_all
    + transform_all
    + trend_all
    + volatility_all
    + volume_all
)

