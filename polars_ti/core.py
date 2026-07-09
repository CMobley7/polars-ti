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
from polars_ti.candles import (
    cdl_2crows,
    cdl_3blackcrows,
    cdl_3inside,
    cdl_3linestrike,
    cdl_3outside,
    cdl_3starsinsouth,
    cdl_3whitesoldiers,
    cdl_abandonedbaby,
    cdl_advanceblock,
    cdl_belthold,
    cdl_breakaway,
    cdl_closingmarubozu,
    cdl_concealbabyswall,
    cdl_counterattack,
    cdl_darkcloudcover,
    cdl_dojistar,
    cdl_dragonflydoji,
    cdl_engulfing,
    cdl_eveningdojistar,
    cdl_eveningstar,
    cdl_gapsidesidewhite,
    cdl_gravestonedoji,
    cdl_hammer,
    cdl_hangingman,
    cdl_harami,
    cdl_haramicross,
    cdl_highwave,
    cdl_hikkake,
    cdl_hikkakemod,
    cdl_homingpigeon,
    cdl_identical3crows,
    cdl_inneck,
    cdl_invertedhammer,
    cdl_kicking,
    cdl_kickingbylength,
    cdl_ladderbottom,
    cdl_longleggeddoji,
    cdl_longline,
    cdl_marubozu,
    cdl_matchinglow,
    cdl_mathold,
    cdl_morningdojistar,
    cdl_morningstar,
    cdl_onneck,
    cdl_piercing,
    cdl_rickshawman,
    cdl_risefall3methods,
    cdl_separatinglines,
    cdl_shootingstar,
    cdl_shortline,
    cdl_spinningtop,
    cdl_stalledpattern,
    cdl_sticksandwich,
    cdl_takuri,
    cdl_tasukigap,
    cdl_thrusting,
    cdl_tristar,
    cdl_unique3river,
    cdl_upsidegap2crows,
    cdl_xsidegap3methods,
)
from polars_ti.candles.cdl_doji import cdl_doji
from polars_ti.candles.cdl_inside import cdl_inside
from polars_ti.candles.cdl_pattern import cdl, cdl_pattern
from polars_ti.candles.cdl_z import cdl_z
from polars_ti.candles.ha import ha
from polars_ti.cycles.dsp import dsp
from polars_ti.cycles.ebsw import ebsw
from polars_ti.cycles.ht_dcperiod import ht_dcperiod
from polars_ti.cycles.ht_dcphase import ht_dcphase
from polars_ti.cycles.ht_phasor import ht_phasor
from polars_ti.cycles.ht_sine import ht_sine
from polars_ti.cycles.ht_trendmode import ht_trendmode
from polars_ti.cycles.msw import msw
from polars_ti.cycles.reflex import reflex
from polars_ti.ma import ma
from polars_ti.maps import EXCHANGE_TZ, Category, Imports, version
from polars_ti.momentum.ao import ao
from polars_ti.momentum.apo import apo
from polars_ti.momentum.bias import bias
from polars_ti.momentum.bop import bop
from polars_ti.momentum.brar import brar
from polars_ti.momentum.cci import cci
from polars_ti.momentum.cfo import cfo
from polars_ti.momentum.cg import cg
from polars_ti.momentum.cmo import cmo
from polars_ti.momentum.coppock import coppock
from polars_ti.momentum.crsi import crsi
from polars_ti.momentum.cti import cti
from polars_ti.momentum.dm import dm
from polars_ti.momentum.er import er
from polars_ti.momentum.eri import eri
from polars_ti.momentum.exhc import exhc
from polars_ti.momentum.fisher import fisher
from polars_ti.momentum.fosc import fosc
from polars_ti.momentum.imi import imi
from polars_ti.momentum.inertia import inertia
from polars_ti.momentum.kdj import kdj
from polars_ti.momentum.kst import kst
from polars_ti.momentum.lrsi import lrsi
from polars_ti.momentum.macd import macd
from polars_ti.momentum.mom import mom
from polars_ti.momentum.pgo import pgo
from polars_ti.momentum.po import po
from polars_ti.momentum.ppo import ppo
from polars_ti.momentum.psl import psl
from polars_ti.momentum.qqe import qqe
from polars_ti.momentum.rmi import rmi
from polars_ti.momentum.roc import roc
from polars_ti.momentum.rocp import rocp
from polars_ti.momentum.rocr import rocr
from polars_ti.momentum.rocr100 import rocr100
from polars_ti.momentum.rsi import rsi
from polars_ti.momentum.rsx import rsx
from polars_ti.momentum.rvgi import rvgi
from polars_ti.momentum.slope import slope
from polars_ti.momentum.smc import smc
from polars_ti.momentum.smc_sweep import smc_sweep
from polars_ti.momentum.smi import smi
from polars_ti.momentum.squeeze import squeeze
from polars_ti.momentum.squeeze_pro import squeeze_pro
from polars_ti.momentum.stc import stc
from polars_ti.momentum.stoch import stoch
from polars_ti.momentum.stochf import stochf
from polars_ti.momentum.stochrsi import stochrsi
from polars_ti.momentum.tmo import tmo
from polars_ti.momentum.trix import trix
from polars_ti.momentum.trixh import trixh
from polars_ti.momentum.tsi import tsi
from polars_ti.momentum.uo import uo
from polars_ti.momentum.vwmacd import vwmacd
from polars_ti.momentum.willr import willr
from polars_ti.overlap.alligator import alligator
from polars_ti.overlap.alma import alma
from polars_ti.overlap.avgprice import avgprice
from polars_ti.overlap.dema import dema
from polars_ti.overlap.ema import ema
from polars_ti.overlap.fwma import fwma
from polars_ti.overlap.hilo import hilo
from polars_ti.overlap.hl2 import hl2
from polars_ti.overlap.hlc3 import hlc3
from polars_ti.overlap.hma import hma
from polars_ti.overlap.hwma import hwma
from polars_ti.overlap.ichimoku import ichimoku
from polars_ti.overlap.jma import jma
from polars_ti.overlap.kama import kama
from polars_ti.overlap.linreg import linreg
from polars_ti.overlap.mama import mama
from polars_ti.overlap.mavp import mavp
from polars_ti.overlap.mcgd import mcgd
from polars_ti.overlap.medprice import medprice
from polars_ti.overlap.midpoint import midpoint
from polars_ti.overlap.midprice import midprice
from polars_ti.overlap.mmar import mmar
from polars_ti.overlap.ohlc4 import ohlc4
from polars_ti.overlap.ott import ott
from polars_ti.overlap.pivots import pivots
from polars_ti.overlap.pwma import pwma
from polars_ti.overlap.rainbow import rainbow
from polars_ti.overlap.rma import rma
from polars_ti.overlap.sinwma import sinwma
from polars_ti.overlap.sma import sma
from polars_ti.overlap.smma import smma
from polars_ti.overlap.ssf import ssf
from polars_ti.overlap.ssf3 import ssf3
from polars_ti.overlap.supertrend import supertrend
from polars_ti.overlap.swma import swma
from polars_ti.overlap.t3 import t3
from polars_ti.overlap.tema import tema
from polars_ti.overlap.trima import trima
from polars_ti.overlap.tsf import tsf
from polars_ti.overlap.typprice import typprice
from polars_ti.overlap.vidya import vidya
from polars_ti.overlap.wcp import wcp
from polars_ti.overlap.wma import wma
from polars_ti.overlap.zlma import zlma
from polars_ti.performance.drawdown import drawdown
from polars_ti.performance.log_return import log_return
from polars_ti.performance.percent_return import percent_return
from polars_ti.statistics.beta import beta
from polars_ti.statistics.correl import correl
from polars_ti.statistics.entropy import entropy
from polars_ti.statistics.kurtosis import kurtosis
from polars_ti.statistics.mad import mad
from polars_ti.statistics.md import md
from polars_ti.statistics.median import median
from polars_ti.statistics.quantile import quantile
from polars_ti.statistics.skew import skew
from polars_ti.statistics.stdev import stdev
from polars_ti.statistics.stderr import stderr
from polars_ti.statistics.tos_stdevall import tos_stdevall
from polars_ti.statistics.variance import variance
from polars_ti.statistics.zscore import zscore
from polars_ti.transform.cube import cube
from polars_ti.transform.ifisher import ifisher
from polars_ti.transform.remap import remap
from polars_ti.trend.adx import adx
from polars_ti.trend.adxr import adxr
from polars_ti.trend.alphatrend import alphatrend
from polars_ti.trend.amat import amat
from polars_ti.trend.aroon import aroon
from polars_ti.trend.chop import chop
from polars_ti.trend.cksp import cksp
from polars_ti.trend.decay import decay
from polars_ti.trend.decreasing import decreasing
from polars_ti.trend.dpo import dpo
from polars_ti.trend.dx import dx
from polars_ti.trend.ht_trendline import ht_trendline
from polars_ti.trend.increasing import increasing
from polars_ti.trend.long_run import long_run
from polars_ti.trend.pmax import pmax
from polars_ti.trend.psar import psar
from polars_ti.trend.qstick import qstick
from polars_ti.trend.rwi import rwi
from polars_ti.trend.short_run import short_run
from polars_ti.trend.trama import trama
from polars_ti.trend.trendflex import trendflex
from polars_ti.trend.tsignals import tsignals
from polars_ti.trend.ttm_trend import ttm_trend
from polars_ti.trend.vhf import vhf
from polars_ti.trend.vortex import vortex
from polars_ti.trend.xsignals import xsignals
from polars_ti.trend.zigzag import zigzag
from polars_ti.volatility.aberration import aberration
from polars_ti.volatility.accbands import accbands
from polars_ti.volatility.atr import atr
from polars_ti.volatility.atrts import atrts
from polars_ti.volatility.avsl import avsl
from polars_ti.volatility.avolume import avolume
from polars_ti.volatility.bbands import bbands
from polars_ti.volatility.chandelier_exit import chandelier_exit
from polars_ti.volatility.cvi import cvi
from polars_ti.volatility.donchian import donchian
from polars_ti.volatility.fvg import fvg
from polars_ti.volatility.halftrend import halftrend
from polars_ti.volatility.hvol import hvol
from polars_ti.volatility.hwc import hwc
from polars_ti.volatility.kc import kc
from polars_ti.volatility.massi import massi
from polars_ti.volatility.natr import natr
from polars_ti.volatility.pdist import pdist
from polars_ti.volatility.rvi import rvi
from polars_ti.volatility.thermo import thermo
from polars_ti.volatility.true_range import true_range
from polars_ti.volatility.ui import ui
from polars_ti.volume.ad import ad
from polars_ti.volume.adosc import adosc
from polars_ti.volume.aobv import aobv
from polars_ti.volume.avwap import avwap
from polars_ti.volume.cmf import cmf
from polars_ti.volume.efi import efi
from polars_ti.volume.emv import emv
from polars_ti.volume.eom import eom
from polars_ti.volume.kvo import kvo
from polars_ti.volume.marketfi import marketfi
from polars_ti.volume.mfi import mfi
from polars_ti.volume.nvi import nvi
from polars_ti.volume.obv import obv
from polars_ti.volume.pvi import pvi
from polars_ti.volume.pvo import pvo
from polars_ti.volume.pvol import pvol
from polars_ti.volume.pvr import pvr
from polars_ti.volume.pvt import pvt
from polars_ti.volume.vfi import vfi
from polars_ti.volume.vhm import vhm
from polars_ti.volume.vosc import vosc
from polars_ti.volume.vp import vp
from polars_ti.volume.vwap import vwap
from polars_ti.volume.vwma import vwma
from polars_ti.volume.wad import wad
from polars_ti.volume.wb_tsv import wb_tsv


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
    ) -> pl.DataFrame | None:
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
            # Indicator returned None (invalid/short input): propagate None rather
            # than silently handing back the entire parent DataFrame.
            return None

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
            "categories",
            "indicators",
            "help",
            "reverse",
            "study",
            "strategy",
            "to_utc",
            "version",
            "exchange",
            "cores",
        }
        user_skip = set(exclude) if exclude else set()
        inds = sorted(name for name in dir(self) if not name.startswith("_") and name not in skip | user_skip)
        if as_list:
            return inds
        print(f"Polars TI v{self.version} — Available indicators:\n  " + ", ".join(inds))
        return inds

    def reverse(self) -> pl.DataFrame:
        """Return the DataFrame in reverse row order."""
        return self._df.reverse()

    def study(self, study, cores: int = 0, talib: bool = False, errors: str = "warn", **kwargs) -> pl.DataFrame:
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
            errors (str): How to surface indicators that fail during the study.

                * ``"warn"`` (default): collect every failing indicator + its
                  exception and emit a single ``warnings.warn`` summary at the
                  end, while still completing the rest of the study.
                * ``"raise"``: re-raise the first indicator failure immediately.
                * ``"ignore"``: silently skip failing indicators (the historical
                  behaviour).
            **kwargs: Additional keyword arguments forwarded to every indicator.

        Returns:
            The DataFrame with all study columns appended.
        """
        import warnings

        from polars_ti.utils._study import Study
        from polars_ti.maps import Category

        if errors not in ("warn", "raise", "ignore"):
            raise ValueError(f"errors must be one of 'warn'|'raise'|'ignore', got {errors!r}")

        # Snapshot the input frame so this study computes from the ORIGINAL
        # columns. study() accumulates its output onto self._df while running;
        # without restoring afterward, a second study on the same accessor would
        # see the first run's columns and skip recomputing them (line in _run
        # that filters `new_cols`), leaking stale values across runs.
        _saved_df = self._df

        # Collected (indicator, exception) pairs for the "warn" summary.
        failures: list[tuple[str, BaseException]] = []

        def _handle_failure(kind: str, exc: BaseException) -> None:
            """Apply the configured error policy to a failed indicator."""
            if errors == "raise":
                raise exc
            if errors == "warn":
                failures.append((kind, exc))
            # "ignore" -> do nothing

        def _run(kind: str, kw: dict) -> None:
            """Dispatch one indicator and hstack new columns onto self._df."""
            fn = getattr(self, kind.lower(), None)
            if fn is None:
                return
            try:
                result = fn(**kw)
            except TypeError as exc:
                if "talib" not in kw or "unexpected keyword argument" not in str(exc):
                    _handle_failure(kind, exc)
                    return
                kw = {k: v for k, v in kw.items() if k != "talib"}
                try:
                    result = fn(**kw)
                except Exception as exc2:
                    _handle_failure(kind, exc2)
                    return
            except Exception as exc:
                _handle_failure(kind, exc)
                return  # Skip indicators that fail (e.g. missing required columns)

            try:
                if isinstance(result, pl.DataFrame) and result.width > 0:
                    new_cols = [c for c in result.columns if c not in self._df.columns]
                    if new_cols:
                        self._df = self._df.hstack(result.select(new_cols))
            except Exception as exc:
                _handle_failure(kind, exc)
                return

        def _finalize() -> pl.DataFrame:
            """Emit the collected-failure summary (warn mode) and return df."""
            if errors == "warn" and failures:
                names = ", ".join(sorted({k for k, _ in failures}))
                detail = "; ".join(f"{k}: {type(e).__name__}: {e}" for k, e in failures)
                warnings.warn(
                    f"study(): {len(failures)} indicator(s) failed and were skipped [{names}]. Details — {detail}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            # Hand back the accumulated study frame, but restore the accessor's
            # working frame to the original input so repeat studies on the same
            # accessor recompute from scratch instead of reusing stale columns.
            result = self._df
            self._df = _saved_df
            return result

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
                kw["talib"] = talib
                _run(kind, kw)
            return _finalize()

        # AllStudy (ti is None) -> run every indicator in every category
        if not isinstance(study, Study) or study.ti is None:
            for category_inds in Category.values():
                for kind in category_inds:
                    kw = dict(kwargs)
                    kw["talib"] = talib
                    _run(kind, kw)
            return _finalize()

        # Custom Study: study.ti is a list of indicator dicts
        for ind_spec in study.ti:
            if not isinstance(ind_spec, dict) or "kind" not in ind_spec:
                continue
            # Shallow-copy so we never mutate the Study definition
            kw = {k: v for k, v in ind_spec.items() if k != "kind"}
            kw.update(kwargs)
            kw["talib"] = talib
            _run(ind_spec["kind"], kw)

        return _finalize()

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
        return self._post_process(cdl_doji(o, h, lo, c, **kw), **kw)

    def cdl_inside(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_inside(o, h, lo, c, **kw), **kw)

    def cdl_pattern(self, name="all", **kw):
        result = cdl_pattern(self._df, name=name, **kw)
        return self._post_process(result, **kw)

    def cdl_2crows(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_2crows(o, h, lo, c, **kw), **kw)

    def cdl_3blackcrows(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_3blackcrows(o, h, lo, c, **kw), **kw)

    def cdl_3inside(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_3inside(o, h, lo, c, **kw), **kw)

    def cdl_3linestrike(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_3linestrike(o, h, lo, c, **kw), **kw)

    def cdl_3outside(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_3outside(o, h, lo, c, **kw), **kw)

    def cdl_3starsinsouth(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_3starsinsouth(o, h, lo, c, **kw), **kw)

    def cdl_3whitesoldiers(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_3whitesoldiers(o, h, lo, c, **kw), **kw)

    def cdl_abandonedbaby(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_abandonedbaby(o, h, lo, c, **kw), **kw)

    def cdl_advanceblock(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_advanceblock(o, h, lo, c, **kw), **kw)

    def cdl_belthold(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_belthold(o, h, lo, c, **kw), **kw)

    def cdl_breakaway(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_breakaway(o, h, lo, c, **kw), **kw)

    def cdl_closingmarubozu(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_closingmarubozu(o, h, lo, c, **kw), **kw)

    def cdl_concealbabyswall(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_concealbabyswall(o, h, lo, c, **kw), **kw)

    def cdl_counterattack(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_counterattack(o, h, lo, c, **kw), **kw)

    def cdl_darkcloudcover(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_darkcloudcover(o, h, lo, c, **kw), **kw)

    def cdl_dojistar(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_dojistar(o, h, lo, c, **kw), **kw)

    def cdl_dragonflydoji(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_dragonflydoji(o, h, lo, c, **kw), **kw)

    def cdl_engulfing(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_engulfing(o, h, lo, c, **kw), **kw)

    def cdl_eveningdojistar(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_eveningdojistar(o, h, lo, c, **kw), **kw)

    def cdl_eveningstar(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_eveningstar(o, h, lo, c, **kw), **kw)

    def cdl_gapsidesidewhite(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_gapsidesidewhite(o, h, lo, c, **kw), **kw)

    def cdl_gravestonedoji(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_gravestonedoji(o, h, lo, c, **kw), **kw)

    def cdl_hammer(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_hammer(o, h, lo, c, **kw), **kw)

    def cdl_hangingman(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_hangingman(o, h, lo, c, **kw), **kw)

    def cdl_harami(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_harami(o, h, lo, c, **kw), **kw)

    def cdl_haramicross(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_haramicross(o, h, lo, c, **kw), **kw)

    def cdl_highwave(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_highwave(o, h, lo, c, **kw), **kw)

    def cdl_hikkake(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_hikkake(o, h, lo, c, **kw), **kw)

    def cdl_hikkakemod(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_hikkakemod(o, h, lo, c, **kw), **kw)

    def cdl_homingpigeon(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_homingpigeon(o, h, lo, c, **kw), **kw)

    def cdl_identical3crows(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_identical3crows(o, h, lo, c, **kw), **kw)

    def cdl_inneck(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_inneck(o, h, lo, c, **kw), **kw)

    def cdl_invertedhammer(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_invertedhammer(o, h, lo, c, **kw), **kw)

    def cdl_kicking(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_kicking(o, h, lo, c, **kw), **kw)

    def cdl_kickingbylength(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_kickingbylength(o, h, lo, c, **kw), **kw)

    def cdl_ladderbottom(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_ladderbottom(o, h, lo, c, **kw), **kw)

    def cdl_longleggeddoji(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_longleggeddoji(o, h, lo, c, **kw), **kw)

    def cdl_longline(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_longline(o, h, lo, c, **kw), **kw)

    def cdl_marubozu(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_marubozu(o, h, lo, c, **kw), **kw)

    def cdl_matchinglow(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_matchinglow(o, h, lo, c, **kw), **kw)

    def cdl_mathold(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_mathold(o, h, lo, c, **kw), **kw)

    def cdl_morningdojistar(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_morningdojistar(o, h, lo, c, **kw), **kw)

    def cdl_morningstar(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_morningstar(o, h, lo, c, **kw), **kw)

    def cdl_onneck(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_onneck(o, h, lo, c, **kw), **kw)

    def cdl_piercing(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_piercing(o, h, lo, c, **kw), **kw)

    def cdl_rickshawman(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_rickshawman(o, h, lo, c, **kw), **kw)

    def cdl_risefall3methods(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_risefall3methods(o, h, lo, c, **kw), **kw)

    def cdl_separatinglines(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_separatinglines(o, h, lo, c, **kw), **kw)

    def cdl_shootingstar(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_shootingstar(o, h, lo, c, **kw), **kw)

    def cdl_shortline(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_shortline(o, h, lo, c, **kw), **kw)

    def cdl_spinningtop(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_spinningtop(o, h, lo, c, **kw), **kw)

    def cdl_stalledpattern(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_stalledpattern(o, h, lo, c, **kw), **kw)

    def cdl_sticksandwich(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_sticksandwich(o, h, lo, c, **kw), **kw)

    def cdl_takuri(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_takuri(o, h, lo, c, **kw), **kw)

    def cdl_tasukigap(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_tasukigap(o, h, lo, c, **kw), **kw)

    def cdl_thrusting(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_thrusting(o, h, lo, c, **kw), **kw)

    def cdl_tristar(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_tristar(o, h, lo, c, **kw), **kw)

    def cdl_unique3river(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_unique3river(o, h, lo, c, **kw), **kw)

    def cdl_upsidegap2crows(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_upsidegap2crows(o, h, lo, c, **kw), **kw)

    def cdl_xsidegap3methods(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_xsidegap3methods(o, h, lo, c, **kw), **kw)

    def cdl_z(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cdl_z(o, h, lo, c, **kw), **kw)

    def ha(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(ha(o, h, lo, c, **kw), **kw)

    # ==================================================================
    #  Cycles
    # ==================================================================

    def dsp(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(dsp(c, **kw), **kw)

    def ebsw(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(ebsw(c, **kw), **kw)

    def reflex(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(reflex(c, **kw), **kw)

    def ht_dcperiod(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(ht_dcperiod(c, **kw), **kw)

    def ht_dcphase(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(ht_dcphase(c, **kw), **kw)

    def ht_phasor(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(ht_phasor(c, **kw), **kw)

    def ht_sine(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(ht_sine(c, **kw), **kw)

    def ht_trendmode(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(ht_trendmode(c, **kw), **kw)

    def msw(self, close=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(msw(c, **kw), **kw)

    # ==================================================================
    #  Momentum
    # ==================================================================

    def ao(self, high=None, low=None, **kw):
        return self._post_process(
            ao(
                self._high(kw) if not high else self._col(high),
                self._low(kw) if not low else self._col(low),
                **kw,
            ),
            **kw,
        )

    def apo(self, close=None, **kw):
        return self._post_process(apo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def bias(self, close=None, **kw):
        return self._post_process(bias(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def bop(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open")
        kw.setdefault("high", "high")
        kw.setdefault("low", "low")
        kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open"))
        h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low"))
        c = self._col(close or kw.pop("close"))
        return self._post_process(bop(o, h, lo, c, **kw), **kw)

    def brar(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open")
        kw.setdefault("high", "high")
        kw.setdefault("low", "low")
        kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open"))
        h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low"))
        c = self._col(close or kw.pop("close"))
        return self._post_process(brar(o, h, lo, c, **kw), **kw)

    def cci(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cci(h, lo, c, **kw), **kw)

    def cfo(self, close=None, **kw):
        return self._post_process(cfo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def cg(self, close=None, **kw):
        return self._post_process(cg(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def cmo(self, close=None, **kw):
        return self._post_process(cmo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def coppock(self, close=None, **kw):
        return self._post_process(coppock(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def crsi(self, close=None, **kw):
        return self._post_process(crsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def cti(self, close=None, **kw):
        return self._post_process(cti(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def dm(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(dm(h, lo, **kw), **kw)

    def er(self, close=None, **kw):
        return self._post_process(er(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def eri(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(eri(h, lo, c, **kw), **kw)

    def exhc(self, close=None, **kw):
        return self._post_process(exhc(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def fisher(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(fisher(h, lo, **kw), **kw)

    def fosc(self, close=None, **kw):
        return self._post_process(fosc(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def imi(self, open_=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(imi(o, c, **kw), **kw)

    def inertia(self, high=None, low=None, close=None, **kw):
        # refined=True / thirds=True modes require high & low; resolve them so
        # those modes don't raise ColumnNotFoundError.
        c = self._col(close or kw.pop("close", "close"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(inertia(c, high=h, low=lo, **kw), **kw)

    def kdj(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(kdj(h, lo, c, **kw), **kw)

    def kst(self, close=None, **kw):
        return self._post_process(kst(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def lrsi(self, close=None, **kw):
        return self._post_process(lrsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def macd(self, close=None, **kw):
        return self._post_process(macd(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def mom(self, close=None, **kw):
        return self._post_process(mom(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def pgo(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pgo(h, lo, c, **kw), **kw)

    def po(self, close=None, **kw):
        return self._post_process(po(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ppo(self, close=None, **kw):
        return self._post_process(ppo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def psl(self, close=None, **kw):
        return self._post_process(psl(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def qqe(self, close=None, **kw):
        return self._post_process(qqe(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rmi(self, close=None, **kw):
        return self._post_process(rmi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def roc(self, close=None, **kw):
        return self._post_process(roc(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rocp(self, close=None, **kw):
        return self._post_process(rocp(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rocr(self, close=None, **kw):
        return self._post_process(rocr(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rocr100(self, close=None, **kw):
        return self._post_process(rocr100(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rsi(self, close=None, **kw):
        return self._post_process(rsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rsx(self, close=None, **kw):
        return self._post_process(rsx(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rvgi(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(rvgi(o, h, lo, c, **kw), **kw)

    def slope(self, close=None, **kw):
        return self._post_process(slope(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def smc(self, open_=None, high=None, low=None, close=None, **kw):
        # smc() signature is smc(open_, high, low, close, ...).
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(smc(o, h, lo, c, **kw), **kw)

    def smc_sweep(self, open_=None, high=None, low=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(smc_sweep(o, h, lo, c, **kw), **kw)

    def smi(self, close=None, **kw):
        return self._post_process(smi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def squeeze(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(squeeze(h, lo, c, **kw), **kw)

    def squeeze_pro(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(squeeze_pro(h, lo, c, **kw), **kw)

    def stc(self, close=None, **kw):
        return self._post_process(stc(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def stoch(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(stoch(h, lo, c, **kw), **kw)

    def stochf(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(stochf(h, lo, c, **kw), **kw)

    def stochrsi(self, close=None, **kw):
        return self._post_process(stochrsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def tmo(self, open_=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(tmo(o, c, **kw), **kw)

    def trix(self, close=None, **kw):
        return self._post_process(trix(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def trixh(self, high=None, low=None, close=None, **kw):
        # trixh() signature is trixh(close, ...); high/low are unused.
        c = self._col(close or kw.pop("close", "close"))
        kw.pop("high", None)
        kw.pop("low", None)
        return self._post_process(trixh(c, **kw), **kw)

    def tsi(self, close=None, **kw):
        return self._post_process(tsi(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def uo(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(uo(h, lo, c, **kw), **kw)

    def vwmacd(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(vwmacd(c, v, **kw), **kw)

    def willr(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(willr(h, lo, c, **kw), **kw)

    # ==================================================================
    #  Overlap
    # ==================================================================

    def alligator(self, close=None, **kw):
        return self._post_process(alligator(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def alma(self, close=None, **kw):
        return self._post_process(alma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def avgprice(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open")
        kw.setdefault("high", "high")
        kw.setdefault("low", "low")
        kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open"))
        h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low"))
        c = self._col(close or kw.pop("close"))
        return self._post_process(avgprice(o, h, lo, c, **kw), **kw)

    def dema(self, close=None, **kw):
        return self._post_process(dema(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ema(self, close=None, **kw):
        return self._post_process(ema(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def fwma(self, close=None, **kw):
        return self._post_process(fwma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def hilo(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(hilo(h, lo, c, **kw), **kw)

    def hl2(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(hl2(h, lo, **kw), **kw)

    def hlc3(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(hlc3(h, lo, c, **kw), **kw)

    def hma(self, close=None, **kw):
        return self._post_process(hma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def hwma(self, close=None, **kw):
        return self._post_process(hwma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ichimoku(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(ichimoku(h, lo, c, **kw), **kw)

    def jma(self, close=None, **kw):
        return self._post_process(jma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def kama(self, close=None, **kw):
        return self._post_process(kama(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def linreg(self, close=None, **kw):
        return self._post_process(linreg(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def mama(self, close=None, **kw):
        return self._post_process(mama(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def mavp(self, close=None, periods=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        p = self._col(periods or kw.pop("periods", "periods"))
        return self._post_process(mavp(c, p, **kw), **kw)

    def mcgd(self, close=None, **kw):
        return self._post_process(mcgd(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def medprice(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(medprice(h, lo, **kw), **kw)

    def midpoint(self, close=None, **kw):
        return self._post_process(midpoint(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def midprice(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(midprice(h, lo, **kw), **kw)

    def mmar(self, close=None, **kw):
        return self._post_process(mmar(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ohlc4(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open")
        kw.setdefault("high", "high")
        kw.setdefault("low", "low")
        kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open"))
        h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low"))
        c = self._col(close or kw.pop("close"))
        return self._post_process(ohlc4(o, h, lo, c, **kw), **kw)

    def ott(self, close=None, **kw):
        return self._post_process(ott(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def pivots(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(pivots(h, lo, c, **kw), **kw)

    def pwma(self, close=None, **kw):
        return self._post_process(pwma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rainbow(self, close=None, **kw):
        return self._post_process(rainbow(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def rma(self, close=None, **kw):
        return self._post_process(rma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def sinwma(self, close=None, **kw):
        return self._post_process(sinwma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def sma(self, close=None, **kw):
        return self._post_process(sma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def smma(self, close=None, **kw):
        return self._post_process(smma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ssf(self, close=None, **kw):
        return self._post_process(ssf(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ssf3(self, close=None, **kw):
        return self._post_process(ssf3(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def supertrend(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(supertrend(h, lo, c, **kw), **kw)

    def swma(self, close=None, **kw):
        return self._post_process(swma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def t3(self, close=None, **kw):
        return self._post_process(t3(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def tema(self, close=None, **kw):
        return self._post_process(tema(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def trima(self, close=None, **kw):
        return self._post_process(trima(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def tsf(self, close=None, **kw):
        return self._post_process(tsf(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def typprice(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(typprice(h, lo, c, **kw), **kw)

    def vidya(self, close=None, **kw):
        return self._post_process(vidya(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def wcp(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(wcp(h, lo, c, **kw), **kw)

    def wma(self, close=None, **kw):
        return self._post_process(wma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def zlma(self, close=None, **kw):
        return self._post_process(zlma(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Performance
    # ==================================================================

    def drawdown(self, close=None, **kw):
        return self._post_process(drawdown(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def log_return(self, close=None, **kw):
        return self._post_process(log_return(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def percent_return(self, close=None, **kw):
        return self._post_process(percent_return(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Statistics
    # ==================================================================

    def beta(self, close=None, benchmark=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        b = self._col(benchmark or kw.pop("benchmark", "benchmark"))
        return self._post_process(beta(c, b, **kw), **kw)

    def correl(self, close=None, benchmark=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        b = self._col(benchmark or kw.pop("benchmark", "benchmark"))
        return self._post_process(correl(c, b, **kw), **kw)

    def entropy(self, close=None, **kw):
        return self._post_process(entropy(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def kurtosis(self, close=None, **kw):
        return self._post_process(kurtosis(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def mad(self, close=None, **kw):
        return self._post_process(mad(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def md(self, close=None, **kw):
        return self._post_process(md(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def median(self, close=None, **kw):
        return self._post_process(median(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def quantile(self, close=None, **kw):
        return self._post_process(quantile(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def skew(self, close=None, **kw):
        return self._post_process(skew(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def stdev(self, close=None, **kw):
        return self._post_process(stdev(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def stderr(self, close=None, **kw):
        return self._post_process(stderr(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def tos_stdevall(self, close=None, **kw):
        return self._post_process(tos_stdevall(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def variance(self, close=None, **kw):
        return self._post_process(variance(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def zscore(self, close=None, **kw):
        return self._post_process(zscore(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Transform
    # ==================================================================

    def cube(self, close=None, **kw):
        return self._post_process(cube(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def ifisher(self, close=None, **kw):
        return self._post_process(ifisher(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def remap(self, close=None, **kw):
        return self._post_process(remap(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Trend
    # ==================================================================

    def adx(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(adx(h, lo, c, **kw), **kw)

    def adxr(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(adxr(h, lo, c, **kw), **kw)

    def alphatrend(self, high=None, low=None, close=None, volume=None, **kw):
        # alphatrend() signature is alphatrend(high, low, close, ...); volume unused.
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        kw.pop("volume", None)
        return self._post_process(alphatrend(h, lo, c, **kw), **kw)

    def amat(self, close=None, **kw):
        return self._post_process(amat(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def aroon(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(aroon(h, lo, **kw), **kw)

    def chop(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(chop(h, lo, c, **kw), **kw)

    def cksp(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(cksp(h, lo, c, **kw), **kw)

    def decay(self, close=None, **kw):
        return self._post_process(decay(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def decreasing(self, close=None, **kw):
        return self._post_process(decreasing(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def dpo(self, close=None, **kw):
        return self._post_process(dpo(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def dx(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(dx(h, lo, c, **kw), **kw)

    def ht_trendline(self, close=None, **kw):
        return self._post_process(ht_trendline(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def increasing(self, close=None, **kw):
        return self._post_process(increasing(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def long_run(self, fast=None, slow=None, **kw):
        f = self._col(fast or kw.pop("fast", "fast"))
        s = self._col(slow or kw.pop("slow", "slow"))
        return self._post_process(long_run(f, s, **kw), **kw)

    def pmax(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pmax(h, lo, c, v, **kw), **kw)

    def psar(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(psar(h, lo, c, **kw), **kw)

    def qstick(self, open_=None, close=None, **kw):
        o = self._col(open_ or kw.pop("open", "open"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(qstick(o, c, **kw), **kw)

    def rwi(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(rwi(h, lo, c, **kw), **kw)

    def short_run(self, fast=None, slow=None, **kw):
        f = self._col(fast or kw.pop("fast", "fast"))
        s = self._col(slow or kw.pop("slow", "slow"))
        return self._post_process(short_run(f, s, **kw), **kw)

    def trama(self, close=None, **kw):
        return self._post_process(trama(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def trendflex(self, close=None, **kw):
        return self._post_process(trendflex(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def tsignals(self, trend=None, **kw):
        t = self._col(trend or kw.pop("trend", "trend"))
        return self._post_process(tsignals(t, **kw), **kw)

    def ttm_trend(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(ttm_trend(h, lo, c, **kw), **kw)

    def vhf(self, close=None, **kw):
        return self._post_process(vhf(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def vortex(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(vortex(h, lo, c, **kw), **kw)

    def xsignals(self, signal=None, xa=None, xb=None, **kw):
        # xa/xb are numeric cross thresholds (or column names), NOT the boolean
        # `above` flag; pass them through unchanged and only resolve `signal`.
        s = self._col(signal or kw.pop("signal", "signal"))
        xa = xa if xa is not None else kw.pop("xa", None)
        xb = xb if xb is not None else kw.pop("xb", None)
        return self._post_process(xsignals(s, xa, xb, **kw), **kw)

    def zigzag(self, high=None, low=None, close=None, **kw):
        # zigzag() signature is zigzag(high, low, ...); 'close' is unused.
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        kw.pop("close", None)
        return self._post_process(zigzag(h, lo, **kw), **kw)

    # ==================================================================
    #  Volatility
    # ==================================================================

    def aberration(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(aberration(h, lo, c, **kw), **kw)

    def accbands(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(accbands(h, lo, c, **kw), **kw)

    def atr(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(atr(h, lo, c, **kw), **kw)

    def atrts(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(atrts(h, lo, c, **kw), **kw)

    def avsl(self, close=None, low=None, volume=None, **kw):
        # avsl() signature is avsl(close, low, volume, ...); 'high' is unused.
        c = self._col(close or kw.pop("close", "close"))
        lo = self._col(low or kw.pop("low", "low"))
        v = self._col(volume or kw.pop("volume", "volume"))
        kw.pop("high", None)
        return self._post_process(avsl(c, lo, v, **kw), **kw)

    def avolume(self, close=None, **kw):
        kw.pop("high", None)
        kw.pop("low", None)
        kw.pop("volume", None)
        return self._post_process(avolume(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def bbands(self, close=None, **kw):
        return self._post_process(bbands(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def chandelier_exit(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(chandelier_exit(h, lo, c, **kw), **kw)

    def cvi(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        kw.pop("close", None)
        kw.pop("volume", None)
        return self._post_process(cvi(h, lo, **kw), **kw)

    def donchian(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(donchian(h, lo, **kw), **kw)

    def fvg(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open")
        kw.setdefault("high", "high")
        kw.setdefault("low", "low")
        kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open"))
        h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low"))
        c = self._col(close or kw.pop("close"))
        return self._post_process(fvg(o, h, lo, c, **kw), **kw)

    def halftrend(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(halftrend(h, lo, c, **kw), **kw)

    def hvol(self, close=None, **kw):
        kw.pop("high", None)
        kw.pop("low", None)
        kw.pop("volume", None)
        return self._post_process(hvol(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def hwc(self, close=None, **kw):
        return self._post_process(hwc(self._col(close or kw.pop("close", "close")), **kw), **kw)

    def kc(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(kc(h, lo, c, **kw), **kw)

    def massi(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(massi(h, lo, **kw), **kw)

    def natr(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(natr(h, lo, c, **kw), **kw)

    def pdist(self, open_=None, high=None, low=None, close=None, **kw):
        kw.setdefault("open", "open")
        kw.setdefault("high", "high")
        kw.setdefault("low", "low")
        kw.setdefault("close", "close")
        o = self._col(open_ or kw.pop("open"))
        h = self._col(high or kw.pop("high"))
        lo = self._col(low or kw.pop("low"))
        c = self._col(close or kw.pop("close"))
        return self._post_process(pdist(o, h, lo, c, **kw), **kw)

    def rvi(self, open_=None, high=None, low=None, close=None, **kw):
        # rvi() signature is rvi(close, high, low, ...); 'open' is unused.
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        kw.pop("open", None)
        return self._post_process(rvi(c, h, lo, **kw), **kw)

    def thermo(self, high=None, low=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        return self._post_process(thermo(h, lo, **kw), **kw)

    def true_range(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        return self._post_process(true_range(h, lo, c, **kw), **kw)

    def ui(self, close=None, **kw):
        return self._post_process(ui(self._col(close or kw.pop("close", "close")), **kw), **kw)

    # ==================================================================
    #  Volume
    # ==================================================================

    def ad(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(ad(h, lo, c, v, **kw), **kw)

    def adosc(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(adosc(h, lo, c, v, **kw), **kw)

    def aobv(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(aobv(c, v, **kw), **kw)

    def avwap(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(avwap(h, lo, c, v, **kw), **kw)

    def cmf(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(cmf(h, lo, c, v, **kw), **kw)

    def efi(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(efi(c, v, **kw), **kw)

    def emv(self, high=None, low=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        v = self._col(volume or kw.pop("volume", "volume"))
        kw.pop("close", None)
        return self._post_process(emv(h, lo, v, **kw), **kw)

    def eom(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(eom(h, lo, c, v, **kw), **kw)

    def kvo(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(kvo(h, lo, c, v, **kw), **kw)

    def marketfi(self, high=None, low=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        v = self._col(volume or kw.pop("volume", "volume"))
        kw.pop("close", None)
        return self._post_process(marketfi(h, lo, v, **kw), **kw)

    def mfi(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(mfi(h, lo, c, v, **kw), **kw)

    def nvi(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(nvi(c, v, **kw), **kw)

    def obv(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(obv(c, v, **kw), **kw)

    def pvi(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pvi(c, v, **kw), **kw)

    def pvo(self, volume=None, **kw):
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pvo(v, **kw), **kw)

    def pvol(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pvol(c, v, **kw), **kw)

    def pvr(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pvr(c, v, **kw), **kw)

    def pvt(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(pvt(c, v, **kw), **kw)

    def vfi(self, high=None, low=None, close=None, volume=None, **kw):
        # vfi() signature is vfi(close, volume, ...); high/low are unused.
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        kw.pop("high", None)
        kw.pop("low", None)
        return self._post_process(vfi(c, v, **kw), **kw)

    def vhm(self, close=None, volume=None, **kw):
        # vhm() signature is vhm(volume, ...); 'close' is unused.
        v = self._col(volume or kw.pop("volume", "volume"))
        kw.pop("close", None)
        return self._post_process(vhm(v, **kw), **kw)

    def vosc(self, volume=None, **kw):
        v = self._col(volume or kw.pop("volume", "volume"))
        kw.pop("high", None)
        kw.pop("low", None)
        kw.pop("close", None)
        return self._post_process(vosc(v, **kw), **kw)

    def vp(self, close=None, volume=None, **kw):
        # vp() needs the DataFrame (uses df.height) and column *names*, like
        # cdl_pattern; passing Exprs raises AttributeError on `.height`.
        c_name = close or kw.pop("close", "close")
        v_name = volume or kw.pop("volume", "volume")
        result = vp(self._df, close=c_name, volume=v_name, **kw)
        return self._post_process(result, **kw)

    def vwap(self, high=None, low=None, close=None, volume=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        # Anchor on a datetime column when one is present (daily reset, like the
        # OLD DatetimeIndex behaviour); otherwise VWAP is cumulative.
        if kw.get("datetime_col") is None:
            for cand in ("date", "datetime", "time", "timestamp"):
                if cand in self._df.columns and self._df.schema[cand] in (pl.Date, pl.Datetime):
                    kw["datetime_col"] = cand
                    break
        return self._post_process(vwap(h, lo, c, v, **kw), **kw)

    def vwma(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(vwma(c, v, **kw), **kw)

    def wad(self, high=None, low=None, close=None, **kw):
        h = self._col(high or kw.pop("high", "high"))
        lo = self._col(low or kw.pop("low", "low"))
        c = self._col(close or kw.pop("close", "close"))
        kw.pop("volume", None)
        return self._post_process(wad(h, lo, c, **kw), **kw)

    def wb_tsv(self, close=None, volume=None, **kw):
        c = self._col(close or kw.pop("close", "close"))
        v = self._col(volume or kw.pop("volume", "volume"))
        return self._post_process(wb_tsv(c, v, **kw), **kw)


# ---------------------------------------------------------------------------
# Wrap every indicator accessor so the DataFrame-level ``append=`` keyword is
# handled centrally instead of leaking into the indicator function (which would
# raise ``TypeError: <ind>() got an unexpected keyword argument 'append'``).
# When ``append=True`` the original DataFrame is returned with the new indicator
# columns hstacked on; otherwise the result-only DataFrame is returned (default).
# ``study``/``categories`` and private methods are excluded.
# ---------------------------------------------------------------------------
import functools as _functools

_NON_INDICATOR_METHODS = {"categories", "study", "strategy"}


def _support_append(method):
    @_functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        append = bool(kwargs.pop("append", False))
        # Universal prefix/suffix support: pop before the indicator fn sees them
        # (it would raise TypeError), then rename produced columns to
        # ``{prefix}_{col}_{suffix}`` (missing parts omitted cleanly). Works for
        # direct calls, append=True, and study() dispatch alike.
        prefix = kwargs.pop("prefix", None)
        suffix = kwargs.pop("suffix", None)
        result = method(self, *args, **kwargs)
        if (prefix or suffix) and isinstance(result, pl.DataFrame):
            pre = f"{prefix}_" if prefix else ""
            suf = f"_{suffix}" if suffix else ""
            result = result.rename({c: f"{pre}{c}{suf}" for c in result.columns})
        if append and isinstance(result, pl.DataFrame):
            new_cols = [c for c in result.columns if c not in self._df.columns]
            return self._df.hstack(result.select(new_cols))
        return result

    return wrapper


for _m_name, _m_attr in list(vars(TechnicalIndicators).items()):
    if _m_name.startswith("_") or _m_name in _NON_INDICATOR_METHODS:
        continue
    if callable(_m_attr):
        setattr(TechnicalIndicators, _m_name, _support_append(_m_attr))
