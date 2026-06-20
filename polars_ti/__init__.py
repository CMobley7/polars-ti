"""
.. moduleauthor:: Christopher Mobley
"""

name = "polars_ti"

from polars_ti.candles import CDL_PATTERN_NAMES
from polars_ti.candles import __all__ as candles_all
from polars_ti.candles import (
    cdl,
    cdl_doji,
    cdl_inside,
    cdl_pattern,
    cdl_z,
    ha,
)

# Enable "ti" DataFrame Extension
from polars_ti.core import TechnicalIndicators

# Study DataClasses — primary API for grouping and running indicator sets
from polars_ti.utils._study import (
    AllStudy,
    AllStrategy,
    CommonStudy,
    CommonStrategy,
    Study,
    Strategy,
)

# Utility functions and numba kernels
from polars_ti import utils
from polars_ti.utils import nb_prenan, nb_prepend, nb_rolling, nb_shift

# Custom External Directory Commands. See help(import_dir)
from polars_ti.custom import create_dir, import_dir
from polars_ti.cycles import __all__ as cycles_all
from polars_ti.cycles import dsp, ebsw, reflex

# Common Averages useful for Indicators
# with a mamode argument, like ti.adx()
from polars_ti.ma import ma

# Dictionaries and version
from polars_ti.maps import EXCHANGE_TZ, RATE, Category, Imports, version
from polars_ti.momentum import __all__ as momentum_all
from polars_ti.momentum import (
    ao,
    apo,
    bias,
    bop,
    brar,
    cci,
    cfo,
    cg,
    cmo,
    coppock,
    crsi,
    cti,
    dm,
    er,
    eri,
    exhc,
    fisher,
    imi,
    inertia,
    kdj,
    kst,
    lrsi,
    macd,
    mom,
    pgo,
    po,
    ppo,
    psl,
    qqe,
    rmi,
    roc,
    rsi,
    rsx,
    rvgi,
    slope,
    smc,
    smi,
    squeeze,
    squeeze_pro,
    stc,
    stoch,
    stochf,
    stochrsi,
    tmo,
    trix,
    trixh,
    tsi,
    uo,
    vwmacd,
    willr,
)
from polars_ti.overlap import __all__ as overlap_all
from polars_ti.overlap import (
    alligator,
    alma,
    dema,
    ema,
    fwma,
    hilo,
    hl2,
    hlc3,
    hma,
    hwma,
    ichimoku,
    jma,
    kama,
    linreg,
    mama,
    mcgd,
    midpoint,
    midprice,
    mmar,
    ohlc4,
    ott,
    pivots,
    pwma,
    rainbow,
    rma,
    sinwma,
    sma,
    smma,
    ssf,
    ssf3,
    supertrend,
    swma,
    t3,
    tema,
    trima,
    vidya,
    wcp,
    wma,
    zlma,
)
from polars_ti.performance import __all__ as performance_all
from polars_ti.performance import drawdown, log_return, percent_return
from polars_ti.statistics import __all__ as statistics_all
from polars_ti.statistics import (
    entropy,
    kurtosis,
    mad,
    median,
    quantile,
    skew,
    stdev,
    tos_stdevall,
    variance,
    zscore,
)
from polars_ti.transform import __all__ as transform_all
from polars_ti.transform import cube, ifisher, remap
from polars_ti.trend import __all__ as trend_all
from polars_ti.trend import (
    adx,
    alphatrend,
    amat,
    aroon,
    chop,
    cksp,
    decay,
    decreasing,
    dpo,
    ht_trendline,
    increasing,
    long_run,
    pmax,
    psar,
    qstick,
    rwi,
    short_run,
    trama,
    trendflex,
    tsignals,
    ttm_trend,
    vhf,
    vortex,
    xsignals,
    zigzag,
)
from polars_ti.volatility import __all__ as volatility_all
from polars_ti.volatility import (
    aberration,
    accbands,
    atr,
    atrts,
    avsl,
    bbands,
    chandelier_exit,
    donchian,
    fvg,
    halftrend,
    hwc,
    kc,
    massi,
    natr,
    pdist,
    rvi,
    thermo,
    true_range,
    ui,
)
from polars_ti.volume import __all__ as volume_all
from polars_ti.volume import (
    ad,
    adosc,
    aobv,
    avwap,
    cmf,
    efi,
    eom,
    kvo,
    mfi,
    nvi,
    obv,
    pvi,
    pvo,
    pvol,
    pvr,
    pvt,
    vfi,
    vhm,
    vp,
    vwap,
    vwma,
    wb_tsv,
)

__all__ = [
    "name",
    "EXCHANGE_TZ",
    "RATE",
    "Category",
    "Imports",
    "version",
    "ma",
    "CDL_PATTERN_NAMES",
    "create_dir",
    "import_dir",
    "TechnicalIndicators",
    # Study API
    "Study",
    "AllStudy",
    "CommonStudy",
    "Strategy",
    "AllStrategy",
    "CommonStrategy",
    # Utilities
    "utils",
    "nb_prenan",
    "nb_prepend",
    "nb_rolling",
    "nb_shift",
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
