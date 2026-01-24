"""
.. moduleauthor:: Christopher Mobley
"""

name = "polars_ti"

from polars_ti.candles import CDL_PATTERN_NAMES
from polars_ti.candles import __all__ as candles_all
from polars_ti.candles import cdl, cdl_doji, cdl_inside, cdl_pattern, cdl_z, ha

# Enable "ti" DataFrame Extension
from polars_ti.core import TechnicalIndicators

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
from polars_ti.utils import (
    AllStrategy,
    AllStudy,
    CommonStrategy,
    CommonStudy,
    Strategy,
    Study,
)
from polars_ti.utils import __all__ as utils_all
from polars_ti.utils import (
    above,
    above_value,
    below,
    below_value,
    cagr,
    calmar_ratio,
    camelCase2Title,
    candle_color,
    category_files,
    combination,
    cross,
    cross_value,
    df_dates,
    df_error_analysis,
    df_month_to_date,
    df_quarter_to_date,
    df_year_to_date,
    downside_deviation,
    erf,
    fibonacci,
    final_time,
    geometric_mean,
    get_time,
    high_low_range,
    hpoly,
    inv_norm,
    is_percent,
    jensens_alpha,
    linear_regression,
    log_geometric_mean,
    log_max_drawdown,
    max_drawdown,
    ms2secs,
    mtd,
    nb_ffill,
    nb_idiff,
    nb_non_zero_range,
    nb_prenan,
    nb_prepend,
    nb_rolling,
    nb_shift,
    non_zero_range,
    optimal_leverage,
    pascals_triangle,
    pure_profit_score,
    qtd,
    real_body,
    recent_maximum_index,
    recent_minimum_index,
    rma_pandas,
    sharpe_ratio,
    signals,
    signed_series,
    simplify_columns,
    sortino_ratio,
    speed_test,
    strided_window,
    symmetric_triangle,
    tal_ma,
    to_utc,
    total_time,
    unix_convert,
    unsigned_differences,
    v_ascending,
    v_bool,
    v_dataframe,
    v_datetime_ordered,
    v_drift,
    v_float,
    v_int,
    v_list,
    v_lowerbound,
    v_mamode,
    v_offset,
    v_pos_default,
    v_scalar,
    v_series,
    v_str,
    v_talib,
    v_tradingview,
    v_upperbound,
    volatility,
    weights,
    ytd,
    zero,
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
    "create_dir",
    "import_dir",
    "TechnicalIndicators",
]

__all__ += [
    utils_all
    + candles_all
    + cycles_all
    + momentum_all
    + overlap_all
    + performance_all
    + statistics_all
    + transform_all
    + trend_all
    + volatility_all
    + volume_all
]
