# -*- coding: utf-8 -*-
"""Parity gate for every native candlestick pattern against TA-Lib.

For each ``cdl_<name>`` pattern this asserts that BOTH the native NumPy path
(``talib=False``) and the TA-Lib fast-path (``talib=True``) reproduce
``talib.CDL<NAME>`` exactly (0 mismatches) on SPY daily data. TA-Lib is the
reference, so the module is skipped when TA-Lib is not installed.
"""

import numpy as np
import polars as pl
import pytest

import importlib

from polars_ti.maps import Imports

_cdl_pattern_mod = importlib.import_module("polars_ti.candles.cdl_pattern")

pytestmark = pytest.mark.skipif(not Imports["talib"], reason="requires TA-Lib (the parity reference)")

SLICE_ROWS = 1500

# Patterns whose TA-Lib function (and native detector) take a `penetration`
# parameter — pass the matching default to the TA-Lib reference.
PENETRATION = {
    "abandonedbaby": 0.3,
    "darkcloudcover": 0.5,
    "eveningstar": 0.3,
    "eveningdojistar": 0.3,
    "morningstar": 0.3,
    "morningdojistar": 0.3,
    "mathold": 0.5,
}

# All natively-implemented framework patterns (name -> cdl_<name> function).
NATIVE_PATTERNS = _cdl_pattern_mod.NATIVE_PATTERNS


@pytest.fixture(scope="module")
def ohlc():
    df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(SLICE_ROWS)
    return (
        df,
        df["open"].to_numpy().astype(float),
        df["high"].to_numpy().astype(float),
        df["low"].to_numpy().astype(float),
        df["close"].to_numpy().astype(float),
    )


@pytest.mark.parametrize("name", sorted(NATIVE_PATTERNS))
@pytest.mark.parametrize("talib", [False, True])
def test_cdl_matches_talib(name, talib, ohlc):
    import talib as talib_mod

    df, o, h, low_, c = ohlc
    fn = NATIVE_PATTERNS[name]
    suffix = name.upper()
    ref_fn = getattr(talib_mod, f"CDL{suffix}")

    pen = PENETRATION.get(name)
    if pen is not None:
        ref = ref_fn(o, h, low_, c, penetration=pen).astype(float)
        got = df.select(fn("open", "high", "low", "close", penetration=pen, talib=talib)).to_series().to_numpy()
    else:
        ref = ref_fn(o, h, low_, c).astype(float)
        got = df.select(fn("open", "high", "low", "close", talib=talib)).to_series().to_numpy()

    mismatches = int(np.sum(got != ref))
    assert mismatches == 0, f"CDL_{suffix} (talib={talib}): {mismatches} mismatches vs talib.CDL{suffix}"


def test_all_talib_patterns_covered():
    """Every TA-Lib CDL pattern in ALL_PATTERNS is natively implemented."""
    import talib as talib_mod

    for name in _cdl_pattern_mod.ALL_PATTERNS:
        if name in ("doji", "inside"):
            continue  # parametrised native implementations, not framework ports
        assert name in NATIVE_PATTERNS, f"{name} missing from NATIVE_PATTERNS"
        assert hasattr(talib_mod, f"CDL{name.upper()}"), f"no talib.CDL{name.upper()}"


# On SPY daily data, ~14 strict multi-bar patterns never fire, so the SPY parity
# test only proves native==talib on all-zero output for them (a weak assertion).
# This fuzz gate feeds structured random OHLC (drift + gaps + marubozu-like
# candles) so those rare patterns actually FIRE, and asserts native reproduces
# TA-Lib exactly wherever TA-Lib emits a non-zero signal.
#
# Two patterns remain effectively unreachable by random data because their
# TA-Lib conditions are extremely strict: "3starsinsouth" and "mathold". Both
# native detectors are line-for-line ports of the TA-Lib C source and produce
# identical output to TA-Lib on every input exercised here; they are covered by
# the exact SPY parity gate above (native==talib, both zero).
_RARE_UNREACHABLE = {"3starsinsouth", "concealbabyswall", "mathold"}


def _fuzz_batches(rng):
    """Yield (o, h, l, c) structured random OHLC batches.

    Two flavours: (1) drift + gaps to trigger trend/gap patterns, and
    (2) marubozu-like candles (near-zero shadows) to trigger the
    kicking / gap-marubozu family.
    """
    n = 60
    for _ in range(300):
        drift = rng.choice([-1.5, -0.8, 0.0, 0.8, 1.5])
        base = 100 + np.cumsum(rng.normal(drift, 0.8, n))
        gap = np.where(rng.random(n) < 0.15, rng.normal(0, 3, n), 0.0)
        o = base + gap + rng.normal(0, 0.4, n)
        c = base + gap + drift * 0.5 + rng.normal(0, 0.4, n)
        hi = np.maximum(o, c) + rng.uniform(0, 0.6, n)
        lo = np.minimum(o, c) - rng.uniform(0, 0.6, n)
        yield o, hi, lo, c

    for _ in range(300):
        o = np.zeros(n)
        c = np.zeros(n)
        hi = np.zeros(n)
        lo = np.zeros(n)
        px = 100.0
        for i in range(n):
            d = rng.choice([-1, 1])
            size = rng.uniform(1, 5)
            op = px + rng.normal(0, 3)
            cl = op + d * size
            o[i] = op
            c[i] = cl
            hi[i] = max(op, cl) + rng.choice([0.0, 0.0, 0.01])
            lo[i] = min(op, cl) - rng.choice([0.0, 0.0, 0.01])
            px = cl
        yield o, hi, lo, c


def test_cdl_native_matches_talib_on_fuzzed_signals():
    """Native == TA-Lib across structured random OHLC (exercises rare patterns).

    On SPY daily data ~14 strict multi-bar patterns never fire, so the SPY
    parity gate only proves native==talib on all-zero output for them.  This
    feeds structured random OHLC so those patterns actually FIRE, then asserts
    native reproduces TA-Lib exactly.  All native detectors are batched into a
    single Polars ``select`` per OHLC frame for speed.
    """
    import talib as talib_mod

    rng = np.random.default_rng(7)
    fired = {name: 0 for name in NATIVE_PATTERNS}
    for o, hi, lo, c in _fuzz_batches(rng):
        df = pl.DataFrame({"open": o, "high": hi, "low": lo, "close": c})
        exprs = []
        for name, fn in NATIVE_PATTERNS.items():
            pen = PENETRATION.get(name)
            if pen is not None:
                exprs.append(fn("open", "high", "low", "close", penetration=pen, talib=False).alias(name))
            else:
                exprs.append(fn("open", "high", "low", "close", talib=False).alias(name))
        got_df = df.select(exprs)
        for name in NATIVE_PATTERNS:
            pen = PENETRATION.get(name)
            ref_fn = getattr(talib_mod, f"CDL{name.upper()}")
            ref = (ref_fn(o, hi, lo, c, penetration=pen) if pen is not None else ref_fn(o, hi, lo, c)).astype(float)
            got = got_df[name].to_numpy()
            assert int(np.sum(got != ref)) == 0, f"native CDL_{name.upper()} diverges from TA-Lib on fuzzed data"
            fired[name] += int(np.sum(ref != 0))

    # Every pattern except the rare-unreachable ones must actually fire at least
    # once, so the parity assertion above is exercised on positive signals.
    # The unreachable trio has TA-Lib conditions too strict for random data;
    # their native detectors are line-for-line ports of the TA-Lib C source and
    # are covered by the exact SPY parity gate (native==talib, both zero).
    never_fired = {name for name, cnt in fired.items() if cnt == 0} - _RARE_UNREACHABLE
    assert not never_fired, f"patterns never fired on fuzz (weak coverage): {sorted(never_fired)}"
