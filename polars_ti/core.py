# -*- coding: utf-8 -*-
"""Polars-TI DataFrame namespace extension.

Registers under ``df.ti`` via ``@pl.api.register_dataframe_namespace("ti")``.

Usage::

    import polars as pl
    import polars_ti  # noqa: F401 — side-effect: registers the 'ti' namespace

    df = pl.read_csv("data.csv")

    # Compute a single indicator (returns a new DataFrame with the result columns)
    sma_df = df.ti.sma()

    # Append result columns to the existing DataFrame
    df = df.ti.sma(append=True)

    # Access indicator with custom parameters
    result = df.ti.macd(fast=12, slow=26, signal=9)
"""
from __future__ import annotations

from multiprocessing import cpu_count

import polars as pl

from polars_ti._typing import DictLike
from polars_ti.candles.cdl_doji import pl_cdl_doji
from polars_ti.candles.cdl_inside import pl_cdl_inside
from polars_ti.candles.cdl_pattern import pl_cdl, pl_cdl_pattern
from polars_ti.candles.cdl_z import pl_cdl_z
from polars_ti.candles.ha import pl_ha
from polars_ti.cycles.dsp import pl_dsp
from polars_ti.cycles.ebsw import pl_ebsw
from polars_ti.cycles.reflex import pl_reflex
from polars_ti.ma import pl_ma
from polars_ti.maps import EXCHANGE_TZ, Category, Imports, version
from polars_ti.momentum.ao import pl_ao
from polars_ti.momentum.apo import pl_apo
from polars_ti.momentum.bias import pl_bias
from polars_ti.momentum.bop import pl_bop
from polars_ti.momentum.brar import pl_brar
from polars_ti.momentum.cci import pl_cci
from polars_ti.momentum.cfo import pl_cfo
from polars_ti.momentum.cg import pl_cg
from polars_ti.momentum.cmo import pl_cmo
from polars_ti.momentum.coppock import pl_coppock
from polars_ti.momentum.crsi import pl_crsi
from polars_ti.momentum.cti import pl_cti
from polars_ti.momentum.dm import pl_dm
from polars_ti.momentum.er import pl_er
from polars_ti.momentum.eri import pl_eri
from polars_ti.momentum.exhc import pl_exhc
from polars_ti.momentum.fisher import pl_fisher
from polars_ti.momentum.imi import pl_imi
from polars_ti.momentum.inertia import pl_inertia
from polars_ti.momentum.kdj import pl_kdj
from polars_ti.momentum.kst import pl_kst
from polars_ti.momentum.lrsi import pl_lrsi
from polars_ti.momentum.macd import pl_macd
from polars_ti.momentum.mom import pl_mom
from polars_ti.momentum.pgo import pl_pgo
from polars_ti.momentum.po import pl_po
from polars_ti.momentum.ppo import pl_ppo
from polars_ti.momentum.psl import pl_psl
from polars_ti.momentum.qqe import pl_qqe
from polars_ti.momentum.rmi import pl_rmi
from polars_ti.momentum.roc import pl_roc
from polars_ti.momentum.rsi import pl_rsi
from polars_ti.momentum.rsx import pl_rsx
from polars_ti.momentum.rvgi import pl_rvgi
from polars_ti.momentum.slope import pl_slope
from polars_ti.momentum.smc import pl_smc
from polars_ti.momentum.smi import pl_smi
from polars_ti.momentum.squeeze import pl_squeeze
from polars_ti.momentum.squeeze_pro import pl_squeeze_pro
from polars_ti.momentum.stc import pl_stc
from polars_ti.momentum.stoch import pl_stoch
from polars_ti.momentum.stochf import pl_stochf
from polars_ti.momentum.stochrsi import pl_stochrsi
from polars_ti.momentum.tmo import pl_tmo
from polars_ti.momentum.trix import pl_trix
from polars_ti.momentum.trixh import pl_trixh
from polars_ti.momentum.tsi import pl_tsi
from polars_ti.momentum.uo import pl_uo
from polars_ti.momentum.vwmacd import pl_vwmacd
from polars_ti.momentum.willr import pl_willr
from polars_ti.overlap.alligator import pl_alligator
from polars_ti.overlap.alma import pl_alma
from polars_ti.overlap.dema import pl_dema
from polars_ti.overlap.ema import pl_ema
from polars_ti.overlap.fwma import pl_fwma
from polars_ti.overlap.hilo import pl_hilo
from polars_ti.overlap.hl2 import pl_hl2
from polars_ti.overlap.hlc3 import pl_hlc3
from polars_ti.overlap.hma import pl_hma
from polars_ti.overlap.hwma import pl_hwma
from polars_ti.overlap.ichimoku import pl_ichimoku
from polars_ti.overlap.jma import pl_jma
from polars_ti.overlap.kama import pl_kama
from polars_ti.overlap.linreg import pl_linreg
from polars_ti.overlap.mama import pl_mama
from polars_ti.overlap.mcgd import pl_mcgd
from polars_ti.overlap.midpoint import pl_midpoint
from polars_ti.overlap.midprice import pl_midprice
from polars_ti.overlap.mmar import pl_mmar
from polars_ti.overlap.ohlc4 import pl_ohlc4
from polars_ti.overlap.ott import pl_ott
from polars_ti.overlap.pivots import pl_pivots
from polars_ti.overlap.pwma import pl_pwma
from polars_ti.overlap.rainbow import pl_rainbow
from polars_ti.overlap.rma import pl_rma
from polars_ti.overlap.sinwma import pl_sinwma
from polars_ti.overlap.sma import pl_sma
from polars_ti.overlap.smma import pl_smma
from polars_ti.overlap.ssf import pl_ssf
from polars_ti.overlap.ssf3 import pl_ssf3
from polars_ti.overlap.supertrend import pl_supertrend
from polars_ti.overlap.swma import pl_swma
from polars_ti.overlap.t3 import pl_t3
from polars_ti.overlap.tema import pl_tema
from polars_ti.overlap.trima import pl_trima
from polars_ti.overlap.vidya import pl_vidya
from polars_ti.overlap.wcp import pl_wcp
from polars_ti.overlap.wma import pl_wma
from polars_ti.overlap.zlma import pl_zlma
from polars_ti.performance.drawdown import pl_drawdown
from polars_ti.performance.log_return import pl_log_return
from polars_ti.performance.percent_return import pl_percent_return
from polars_ti.statistics.entropy import pl_entropy
from polars_ti.statistics.kurtosis import pl_kurtosis
from polars_ti.statistics.mad import pl_mad
from polars_ti.statistics.median import pl_median
from polars_ti.statistics.quantile import pl_quantile
from polars_ti.statistics.skew import pl_skew
from polars_ti.statistics.stdev import pl_stdev
from polars_ti.statistics.tos_stdevall import pl_tos_stdevall
from polars_ti.statistics.variance import pl_variance
from polars_ti.statistics.zscore import pl_zscore
from polars_ti.transform.cube import pl_cube
from polars_ti.transform.ifisher import pl_ifisher
from polars_ti.transform.remap import pl_remap
from polars_ti.trend.adx import pl_adx
from polars_ti.trend.alphatrend import pl_alphatrend
from polars_ti.trend.amat import pl_amat
from polars_ti.trend.aroon import pl_aroon
from polars_ti.trend.chop import pl_chop
from polars_ti.trend.cksp import pl_cksp
from polars_ti.trend.decay import pl_decay
from polars_ti.trend.decreasing import pl_decreasing
from polars_ti.trend.dpo import pl_dpo
from polars_ti.trend.ht_trendline import pl_ht_trendline
from polars_ti.trend.increasing import pl_increasing
from polars_ti.trend.long_run import pl_long_run
from polars_ti.trend.pmax import pl_pmax
from polars_ti.trend.psar import pl_psar
from polars_ti.trend.qstick import pl_qstick
from polars_ti.trend.rwi import pl_rwi
from polars_ti.trend.short_run import pl_short_run
from polars_ti.trend.trama import pl_trama
from polars_ti.trend.trendflex import pl_trendflex
from polars_ti.trend.tsignals import pl_tsignals
from polars_ti.trend.ttm_trend import pl_ttm_trend
from polars_ti.trend.vhf import pl_vhf
from polars_ti.trend.vortex import pl_vortex
from polars_ti.trend.xsignals import pl_xsignals
from polars_ti.trend.zigzag import pl_zigzag
from polars_ti.volatility.aberration import pl_aberration
from polars_ti.volatility.accbands import pl_accbands
from polars_ti.volatility.atr import pl_atr
from polars_ti.volatility.atrts import pl_atrts
from polars_ti.volatility.avsl import pl_avsl
from polars_ti.volatility.bbands import pl_bbands
from polars_ti.volatility.chandelier_exit import pl_chandelier_exit
from polars_ti.volatility.donchian import pl_donchian
from polars_ti.volatility.fvg import pl_fvg
from polars_ti.volatility.halftrend import pl_halftrend
from polars_ti.volatility.hwc import pl_hwc
from polars_ti.volatility.kc import pl_kc
from polars_ti.volatility.massi import pl_massi
from polars_ti.volatility.natr import pl_natr
from polars_ti.volatility.pdist import pl_pdist
from polars_ti.volatility.rvi import pl_rvi
from polars_ti.volatility.thermo import pl_thermo
from polars_ti.volatility.true_range import pl_true_range
from polars_ti.volatility.ui import pl_ui
from polars_ti.volume.ad import pl_ad
from polars_ti.volume.adosc import pl_adosc
from polars_ti.volume.aobv import pl_aobv
from polars_ti.volume.avwap import pl_avwap
from polars_ti.volume.cmf import pl_cmf
from polars_ti.volume.efi import pl_efi
from polars_ti.volume.eom import pl_eom
from polars_ti.volume.kvo import pl_kvo
from polars_ti.volume.mfi import pl_mfi
from polars_ti.volume.nvi import pl_nvi
from polars_ti.volume.obv import pl_obv
from polars_ti.volume.pvi import pl_pvi
from polars_ti.volume.pvo import pl_pvo
from polars_ti.volume.pvol import pl_pvol
from polars_ti.volume.pvr import pl_pvr
from polars_ti.volume.pvt import pl_pvt
from polars_ti.volume.vfi import pl_vfi
from polars_ti.volume.vhm import pl_vhm
from polars_ti.volume.vp import pl_vp
from polars_ti.volume.vwap import pl_vwap
from polars_ti.volume.vwma import pl_vwma
from polars_ti.volume.wb_tsv import pl_wb_tsv


def _collect(result) -> pl.DataFrame | None:
    """Collect either a LazyFrame, one or more Expr, or a DataFrame to raw data."""
    if result is None:
        return None
    if isinstance(result, pl.LazyFrame):
        return result.collect()
    if isinstance(result, pl.DataFrame):
        return result
    # Single or list of pl.Expr — evaluate against a zero-row frame isn't possible
    # without the parent frame; callers that need raw eval should pass it through
    return result


@pl.api.register_dataframe_namespace("ti")
class TechnicalIndicators:
    """Polars DataFrame namespace extension for Technical Indicators.

    Provides a ``df.ti`` accessor that mirrors the behaviour of the legacy
    pandas ``df.ti`` extension, updated for the native Polars API.

    All indicator methods:
    - Accept keyword arguments that override the default column names
      (e.g. ``close="Close"``, ``high="High"``, …)
    - Return a **new** ``pl.DataFrame`` containing only the result columns,
      unless ``append=True`` is passed — in which case the result columns
      are appended to ``self._df`` and the *full* DataFrame is returned.

    Example::

        import polars as pl
        import polars_ti  # registers 'ti' namespace

        df = pl.read_csv("SPY.csv")
        sma20 = df.ti.sma(length=20)                   # new 1-col DataFrame
        df    = df.ti.sma(length=20, append=True)       # df + SMA_20 column
        df    = df.ti(kind="sma", length=20, append=True)
    """

    _cores: int = cpu_count()
    _exchange: str = "NYSE"

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __call__(self, kind: str = None, **kwargs: DictLike):
        """Call an indicator by name: ``df.ti(kind="sma", length=20)``."""
        if isinstance(kind, str):
            fn = getattr(self, kind.lower(), None)
            if fn is None:
                raise AttributeError(f"Unknown indicator: '{kind}'")
            return fn(**kwargs)
        self.help()

    def help(self) -> None:
        """Print available indicators."""
        inds = self.indicators(as_list=True)
        print(f"Polars TI v{self.version} — {len(inds)} indicators:\n  " + ", ".join(inds))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return version

    @property
    def exchange(self) -> str:
        return self._exchange

    @exchange.setter
    def exchange(self, value: str) -> None:
        if value in EXCHANGE_TZ:
            self._exchange = value

    @property
    def cores(self) -> int:
        return self._cores

    @cores.setter
    def cores(self, value: int) -> None:
        cpus = cpu_count()
        self._cores = int(value) if isinstance(value, int) and 0 <= value <= cpus else cpus

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _col(self, name: str) -> pl.Expr:
        """Return ``pl.col(name)`` if the column exists, else raise."""
        if name not in self._df.columns:
            # Try case-insensitive match
            lower = {c.lower(): c for c in self._df.columns}
            if name.lower() in lower:
                return pl.col(lower[name.lower()])
            raise KeyError(f"Column '{name}' not found. Available: {self._df.columns}")
        return pl.col(name)

    def _post_process(
        self,
        exprs,
        append: bool = False,
        **kwargs,
    ) -> pl.DataFrame:
        """Evaluate *exprs* against the parent DataFrame and optionally append.

        Args:
            exprs: A single ``pl.Expr``, a list of ``pl.Expr``, or a
                   ``pl.DataFrame`` already evaluated.
            append: When True, hstack the result columns onto ``self._df``.

        Returns:
            Either the result-only DataFrame or (when append=True) the
            parent DataFrame with result columns hstacked.
        """
        df = self._df

        # Normalise to list[Expr] or DataFrame
        if isinstance(exprs, pl.Expr):
            result_df = df.select(exprs)
        elif isinstance(exprs, list):
            # Flatten: may contain Expr or nested lists (e.g. from pl_exhc)
            flat: list[pl.Expr] = []
            for e in exprs:
                if isinstance(e, list):
                    flat.extend(e)
                else:
                    flat.append(e)
            result_df = df.select(flat)
        elif isinstance(exprs, pl.DataFrame):
            result_df = exprs
        elif isinstance(exprs, pl.LazyFrame):
            result_df = exprs.collect()
        else:
            return df

        if append:
            # Drop columns that already exist (avoid duplicates)
            new_cols = [c for c in result_df.columns if c not in df.columns]
            self._df = df.hstack(result_df.select(new_cols))
            return self._df

        return result_df

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def categories(self) -> list[str]:
        """Return the list of indicator categories."""
        return list(Category.keys())

    def indicators(self, as_list: bool = False, exclude: list[str] | None = None) -> list[str]:
        """Return or print the list of available indicators."""
        skip = {
            "categories", "indicators", "help", "reverse", "study", "strategy",
            "to_utc", "version", "exchange", "cores",
        }
        user_skip = set(exclude) if exclude else set()
        inds = sorted(
            name for name in dir(self)
            if not name.startswith("_") and name not in skip | user_skip
        )
        if as_list:
            return inds
        print(f"Polars TI v{self.version} — Available indicators:\n  " + ", ".join(inds))
        return inds

    def reverse(self) -> pl.DataFrame:
        """Return the DataFrame in reverse row order."""
        return self._df.reverse()

    def study(self, study, cores: int = 0, talib: bool = False, **kwargs) -> pl.DataFrame:
        """Run a :class:`~polars_ti.Study` against this DataFrame.

        Mirrors the original pandas-ti ``df.ti.study()`` API. Each indicator
        dict in ``study.ti`` is dispatched to the matching ``df.ti.<kind>``
        method with ``append=True``.  The *all* study (``study.ti is None``)
        runs every registered indicator.

        Args:
            study: A :class:`~polars_ti.Study` instance, or a category string
                (e.g. ``"momentum"``), or a ``type`` reference such as
                ``ti.AllStudy`` / ``ti.CommonStudy``.
            cores (int): Reserved for future multiprocessing support.
                Currently ignored; all indicators run sequentially.
            talib (bool): Pass ``talib=True`` to each indicator call.
            **kwargs: Additional keyword arguments forwarded to every indicator.

        Returns:
            The DataFrame with all study columns appended in-place via
            ``app        """
        from polars_ti.utils._study import Study
        from polars_ti.maps import Category

        def _run(kind: str, kw: dict) -> None:
            """Dispatch one indicator and hstack new columns onto self._df."""
            fn = getattr(self, kind.lower(), None)
            if fn is None:
                return
            try:
                result = fn(**kw)
                if isinstance(result, pl.DataFrame) and result.width > 0:
                    new_cols = [c for c in result.columns if c not in self._df.columns]
                    if new_cols:
                        self._df = self._df.hstack(result.select(new_cols))
            except Exception:
                pass  # Skip indicators that fail (e.g. missing required columns)

        # Accept a Study class/instance, a category string, or AllStudy sentinel
        if isinstance(study, type) and issubclass(study, Study):
            study = study()  # instantiate if a class was passed

        if isinstance(study, str):
            # Category shorthand: "momentum", "overlap", etc.
            category = study.lower()
            if category not in Category:
                raise ValueError(f"Unknown category '{category}'. Valid: {list(Category.keys())}")
            for kind in Category[category]:
                kw = dict(kwargs)
                if talib:
                    kw["talib"] = True
                _run(kind, kw)
            return self._df

        # AllStudy (ti is None) -> run every indicator in every category
        if not isinstance(study, Study) or study.ti is None:
            for category_inds in Category.values():
                for kind in category_inds:
                    kw = dict(kwargs)
                    if talib:
                        kw["talib"] = True
                    _run(kind, kw)
            return self._df

        # Custom Study: study.ti is a list of indicator dicts
        for ind_spec in study.ti:
            if not isinstance(ind_spec, dict) or "kind" not in ind_spec:
                continue
            # Shallow-copy so we never mutate the Study definition
            kw = {k: v for k, v in ind_spec.items() if k != "kind"}
            kw.update(kwargs)
            if talib:
                kw["talib"] = True
            _run(ind_spec["kind"], kw)

        return self._df

    # Alias for backwards-compatibility with the original pandas-ti API
    strategy = study

    # ------------------------------------------------------------------
    # Helpers for repeated OHLCV column extraction
    # ------------------------------------------------------------------

    def _open(self, kw: dict) -> pl.Expr:
        return self._col(kw.pop("open", "open"))

    def _high(self, kw: dict) -> pl.Expr:
        return self._col(kw.pop("high", "high"))

    def _low(self, kw: dict) -> pl.Expr:
        return self._col(kw.pop("low", "low"))

    def _close(self, kw: dict) -> pl.Expr:
        return self._col(kw.pop("close", "close"))

    def _volume(self, kw: dict) -> pl.Expr:
        return self._col(kw.pop("volume", "volume"))

    # ==================================================================
    #  Candles
    # ==================================================================

    def cdl_doji(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_cdl_doji(o, h, lo, c, **kw), **kw)

    def cdl_inside(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(pl_cdl_inside(h, lo, **kw), **kw)

    def cdl_pattern(self, name="all", **kw):
        result = pl_cdl_pattern(self._df, name=name, **kw)
        return self._post_process(result, **kw)

    def cdl_z(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_cdl_z(o, h, lo, c, **kw), **kw)

    def ha(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_ha(o, h, lo, c, **kw), **kw)

    # ==================================================================
    #  Cycles
    # ==================================================================

    def dsp(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_dsp(c, **kw), **kw)

    def ebsw(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_ebsw(c, **kw), **kw)

    def reflex(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_reflex(c, **kw), **kw)

    # ==================================================================
    #  Momentum
    # ==================================================================

    def ao(self, high=None, low=None, **kw):
        return self._post_process(pl_ao(self._high(kw) if not high else self._col(high),
                                        self._low(kw) if not low else self._col(low), **kw), **kw)

    def apo(self, close=None, **kw):
        return self._post_process(pl_apo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def bias(self, close=None, **kw):
        return self._post_process(pl_bias(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def bop(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open"); kw.setdefault("high", "high")
        kw.setdefault("low", "low"); kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open")); h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low")); c = self._col(close or kw.pop("close"))
        return self._post_process(pl_bop(o, h, lo, c, **kw), **kw)

    def brar(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open"); kw.setdefault("high", "high")
        kw.setdefault("low", "low"); kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open")); h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low")); c = self._col(close or kw.pop("close"))
        return self._post_process(pl_brar(o, h, lo, c, **kw), **kw)

    def cci(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_cci(h, lo, c, **kw), **kw)

    def cfo(self, close=None, **kw):
        return self._post_process(pl_cfo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def cg(self, close=None, **kw):
        return self._post_process(pl_cg(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def cmo(self, close=None, **kw):
        return self._post_process(pl_cmo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def coppock(self, close=None, **kw):
        return self._post_process(pl_coppock(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def crsi(self, close=None, **kw):
        return self._post_process(pl_crsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def cti(self, close=None, **kw):
        return self._post_process(pl_cti(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def dm(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(pl_dm(h, lo, **kw), **kw)

    def er(self, close=None, **kw):
        return self._post_process(pl_er(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def eri(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_eri(h, lo, c, **kw), **kw)

    def exhc(self, close=None, **kw):
        return self._post_process(pl_exhc(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def fisher(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(pl_fisher(h, lo, **kw), **kw)

    def imi(self, open_=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_imi(o, c, **kw), **kw)

    def inertia(self, close=None, **kw):
        return self._post_process(pl_inertia(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def kdj(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_kdj(h, lo, c, **kw), **kw)

    def kst(self, close=None, **kw):
        return self._post_process(pl_kst(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def lrsi(self, close=None, **kw):
        return self._post_process(pl_lrsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def macd(self, close=None, **kw):
        return self._post_process(pl_macd(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def mom(self, close=None, **kw):
        return self._post_process(pl_mom(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def pgo(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_pgo(h, lo, c, **kw), **kw)

    def po(self, close=None, **kw):
        return self._post_process(pl_po(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ppo(self, close=None, **kw):
        return self._post_process(pl_ppo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def psl(self, close=None, **kw):
        return self._post_process(pl_psl(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def qqe(self, close=None, **kw):
        return self._post_process(pl_qqe(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rmi(self, close=None, **kw):
        return self._post_process(pl_rmi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def roc(self, close=None, **kw):
        return self._post_process(pl_roc(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rsi(self, close=None, **kw):
        return self._post_process(pl_rsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rsx(self, close=None, **kw):
        return self._post_process(pl_rsx(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rvgi(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_rvgi(o, h, lo, c, **kw), **kw)

    def slope(self, close=None, **kw):
        return self._post_process(pl_slope(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def smc(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_smc(h, lo, c, **kw), **kw)

    def smi(self, close=None, **kw):
        return self._post_process(pl_smi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def squeeze(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_squeeze(h, lo, c, **kw), **kw)

    def squeeze_pro(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_squeeze_pro(h, lo, c, **kw), **kw)

    def stc(self, close=None, **kw):
        return self._post_process(pl_stc(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def stoch(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_stoch(h, lo, c, **kw), **kw)

    def stochf(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_stochf(h, lo, c, **kw), **kw)

    def stochrsi(self, close=None, **kw):
        return self._post_process(pl_stochrsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def tmo(self, open_=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_tmo(o, c, **kw), **kw)

    def trix(self, close=None, **kw):
        return self._post_process(pl_trix(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def trixh(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_trixh(h, lo, c, **kw), **kw)

    def tsi(self, close=None, **kw):
        return self._post_process(pl_tsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def uo(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_uo(h, lo, c, **kw), **kw)

    def vwmacd(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_vwmacd(c, v, **kw), **kw)

    def willr(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_willr(h, lo, c, **kw), **kw)

    # ==================================================================
    #  Overlap
    # ==================================================================

    def alligator(self, close=None, **kw):
        return self._post_process(pl_alligator(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def alma(self, close=None, **kw):
        return self._post_process(pl_alma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def dema(self, close=None, **kw):
        return self._post_process(pl_dema(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ema(self, close=None, **kw):
        return self._post_process(pl_ema(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def fwma(self, close=None, **kw):
        return self._post_process(pl_fwma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def hilo(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_hilo(h, lo, c, **kw), **kw)

    def hl2(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(pl_hl2(h, lo, **kw), **kw)

    def hlc3(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_hlc3(h, lo, c, **kw), **kw)

    def hma(self, close=None, **kw):
        return self._post_process(pl_hma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def hwma(self, close=None, **kw):
        return self._post_process(pl_hwma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ichimoku(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_ichimoku(h, lo, c, **kw), **kw)

    def jma(self, close=None, **kw):
        return self._post_process(pl_jma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def kama(self, close=None, **kw):
        return self._post_process(pl_kama(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def linreg(self, close=None, **kw):
        return self._post_process(pl_linreg(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def mama(self, close=None, **kw):
        return self._post_process(pl_mama(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def mcgd(self, close=None, **kw):
        return self._post_process(pl_mcgd(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def midpoint(self, close=None, **kw):
        return self._post_process(pl_midpoint(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def midprice(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(pl_midprice(h, lo, **kw), **kw)

    def mmar(self, close=None, **kw):
        return self._post_process(pl_mmar(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ohlc4(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open"); kw.setdefault("high", "high")
        kw.setdefault("low", "low"); kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open")); h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low")); c = self._col(close or kw.pop("close"))
        return self._post_process(pl_ohlc4(o, h, lo, c, **kw), **kw)

    def ott(self, close=None, **kw):
        return self._post_process(pl_ott(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def pivots(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_pivots(h, lo, c, **kw), **kw)

    def pwma(self, close=None, **kw):
        return self._post_process(pl_pwma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rainbow(self, close=None, **kw):
        return self._post_process(pl_rainbow(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rma(self, close=None, **kw):
        return self._post_process(pl_rma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def sinwma(self, close=None, **kw):
        return self._post_process(pl_sinwma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def sma(self, close=None, **kw):
        return self._post_process(pl_sma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def smma(self, close=None, **kw):
        return self._post_process(pl_smma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ssf(self, close=None, **kw):
        return self._post_process(pl_ssf(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ssf3(self, close=None, **kw):
        return self._post_process(pl_ssf3(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def supertrend(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_supertrend(h, lo, c, **kw), **kw)

    def swma(self, close=None, **kw):
        return self._post_process(pl_swma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def t3(self, close=None, **kw):
        return self._post_process(pl_t3(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def tema(self, close=None, **kw):
        return self._post_process(pl_tema(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def trima(self, close=None, **kw):
        return self._post_process(pl_trima(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def vidya(self, close=None, **kw):
        return self._post_process(pl_vidya(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def wcp(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_wcp(h, lo, c, **kw), **kw)

    def wma(self, close=None, **kw):
        return self._post_process(pl_wma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def zlma(self, close=None, **kw):
        return self._post_process(pl_zlma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Performance
    # ==================================================================

    def drawdown(self, close=None, **kw):
        return self._post_process(pl_drawdown(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def log_return(self, close=None, **kw):
        return self._post_process(pl_log_return(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def percent_return(self, close=None, **kw):
        return self._post_process(pl_percent_return(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Statistics
    # ==================================================================

    def entropy(self, close=None, **kw):
        return self._post_process(pl_entropy(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def kurtosis(self, close=None, **kw):
        return self._post_process(pl_kurtosis(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def mad(self, close=None, **kw):
        return self._post_process(pl_mad(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def median(self, close=None, **kw):
        return self._post_process(pl_median(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def quantile(self, close=None, **kw):
        return self._post_process(pl_quantile(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def skew(self, close=None, **kw):
        return self._post_process(pl_skew(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def stdev(self, close=None, **kw):
        return self._post_process(pl_stdev(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def tos_stdevall(self, close=None, **kw):
        return self._post_process(pl_tos_stdevall(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def variance(self, close=None, **kw):
        return self._post_process(pl_variance(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def zscore(self, close=None, **kw):
        return self._post_process(pl_zscore(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Transform
    # ==================================================================

    def cube(self, close=None, **kw):
        return self._post_process(pl_cube(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ifisher(self, close=None, **kw):
        return self._post_process(pl_ifisher(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def remap(self, close=None, **kw):
        return self._post_process(pl_remap(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Trend
    # ==================================================================

    def adx(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_adx(h, lo, c, **kw), **kw)

    def alphatrend(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_alphatrend(h, lo, c, v, **kw), **kw)

    def amat(self, close=None, **kw):
        return self._post_process(pl_amat(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def aroon(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(pl_aroon(h, lo, **kw), **kw)

    def chop(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_chop(h, lo, c, **kw), **kw)

    def cksp(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_cksp(h, lo, c, **kw), **kw)

    def decay(self, close=None, **kw):
        return self._post_process(pl_decay(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def decreasing(self, close=None, **kw):
        return self._post_process(pl_decreasing(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def dpo(self, close=None, **kw):
        return self._post_process(pl_dpo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ht_trendline(self, close=None, **kw):
        return self._post_process(pl_ht_trendline(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def increasing(self, close=None, **kw):
        return self._post_process(pl_increasing(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def long_run(self, fast=None, slow=None, **kw):
        f = self._col(fast or kw.pop("fast", "fast"))
        s = self._col(slow or kw.pop("slow", "slow"))
        return self._post_process(pl_long_run(f, s, **kw), **kw)

    def pmax(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_pmax(h, lo, c, v, **kw), **kw)

    def psar(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_psar(h, lo, c, **kw), **kw)

    def qstick(self, open_=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_qstick(o, c, **kw), **kw)

    def rwi(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_rwi(h, lo, c, **kw), **kw)

    def short_run(self, fast=None, slow=None, **kw):
        f = self._col(fast or kw.pop("fast", "fast"))
        s = self._col(slow or kw.pop("slow", "slow"))
        return self._post_process(pl_short_run(f, s, **kw), **kw)

    def trama(self, close=None, **kw):
        return self._post_process(pl_trama(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def trendflex(self, close=None, **kw):
        return self._post_process(pl_trendflex(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def tsignals(self, trend=None, **kw):
        t = self._col(trend or kw.pop("trend", "trend"))
        return self._post_process(pl_tsignals(t, **kw), **kw)

    def ttm_trend(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_ttm_trend(h, lo, c, **kw), **kw)

    def vhf(self, close=None, **kw):
        return self._post_process(pl_vhf(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def vortex(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_vortex(h, lo, c, **kw), **kw)

    def xsignals(self, signal=None, xa=None, xb=None, **kw):
        s = self._col(signal or kw.pop("signal", "signal"))
        a = self._col(xa or kw.pop("above", "above"))
        b = self._col(xb or kw.pop("below", "below"))
        return self._post_process(pl_xsignals(s, a, b, **kw), **kw)

    def zigzag(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_zigzag(h, lo, c, **kw), **kw)

    # ==================================================================
    #  Volatility
    # ==================================================================

    def aberration(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_aberration(h, lo, c, **kw), **kw)

    def accbands(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_accbands(h, lo, c, **kw), **kw)

    def atr(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_atr(h, lo, c, **kw), **kw)

    def atrts(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_atrts(h, lo, c, **kw), **kw)

    def avsl(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_avsl(h, lo, c, **kw), **kw)

    def bbands(self, close=None, **kw):
        return self._post_process(pl_bbands(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def chandelier_exit(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_chandelier_exit(h, lo, c, **kw), **kw)

    def donchian(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(pl_donchian(h, lo, **kw), **kw)

    def fvg(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open"); kw.setdefault("high", "high")
        kw.setdefault("low", "low"); kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open")); h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low")); c = self._col(close or kw.pop("close"))
        return self._post_process(pl_fvg(o, h, lo, c, **kw), **kw)

    def halftrend(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_halftrend(h, lo, c, **kw), **kw)

    def hwc(self, close=None, **kw):
        return self._post_process(pl_hwc(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def kc(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_kc(h, lo, c, **kw), **kw)

    def massi(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(pl_massi(h, lo, **kw), **kw)

    def natr(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_natr(h, lo, c, **kw), **kw)

    def pdist(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open"); kw.setdefault("high", "high")
        kw.setdefault("low", "low"); kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open")); h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low")); c = self._col(close or kw.pop("close"))
        return self._post_process(pl_pdist(o, h, lo, c, **kw), **kw)

    def rvi(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open"); kw.setdefault("high", "high")
        kw.setdefault("low", "low"); kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open")); h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low")); c = self._col(close or kw.pop("close"))
        return self._post_process(pl_rvi(o, h, lo, c, **kw), **kw)

    def thermo(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(pl_thermo(h, lo, **kw), **kw)

    def true_range(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pl_true_range(h, lo, c, **kw), **kw)

    def ui(self, close=None, **kw):
        return self._post_process(pl_ui(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Volume
    # ==================================================================

    def ad(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_ad(h, lo, c, v, **kw), **kw)

    def adosc(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_adosc(h, lo, c, v, **kw), **kw)

    def aobv(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_aobv(c, v, **kw), **kw)

    def avwap(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_avwap(h, lo, c, v, **kw), **kw)

    def cmf(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_cmf(h, lo, c, v, **kw), **kw)

    def efi(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_efi(c, v, **kw), **kw)

    def eom(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_eom(h, lo, c, v, **kw), **kw)

    def kvo(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_kvo(h, lo, c, v, **kw), **kw)

    def mfi(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_mfi(h, lo, c, v, **kw), **kw)

    def nvi(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_nvi(c, v, **kw), **kw)

    def obv(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_obv(c, v, **kw), **kw)

    def pvi(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_pvi(c, v, **kw), **kw)

    def pvo(self, volume=None, **kw):
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_pvo(v, **kw), **kw)

    def pvol(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_pvol(c, v, **kw), **kw)

    def pvr(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_pvr(c, v, **kw), **kw)

    def pvt(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_pvt(c, v, **kw), **kw)

    def vfi(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_vfi(h, lo, c, v, **kw), **kw)

    def vhm(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_vhm(c, v, **kw), **kw)

    def vp(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_vp(c, v, **kw), **kw)

    def vwap(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_vwap(h, lo, c, v, **kw), **kw)

    def vwma(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_vwma(c, v, **kw), **kw)

    def wb_tsv(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pl_wb_tsv(c, v, **kw), **kw)
