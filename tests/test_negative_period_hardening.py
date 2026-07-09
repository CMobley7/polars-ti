# -*- coding: utf-8 -*-
"""Regression tests for the negative-period hardening pass.

A negative integer period parameter (length/fast/slow/signal/drift/smooth/…) used
to reach unguarded Numba kernels and crash the interpreter with SIGSEGV/SIGABRT
(heap out-of-bounds indexing). The shared ``v_pos_int`` validator now fails fast
with a ``ValueError`` before any kernel runs, so an invalid period is a clean,
catchable error rather than a process crash.

Each case below reproduces a formerly-crashing (indicator, parameter) pair and
asserts it now raises ``ValueError``. A negative value that reached the kernel
would SIGSEGV and take pytest down with it, so the mere fact these run in-process
and assert cleanly is the regression guard.
"""

import polars as pl
import pytest

import polars_ti as ti  # noqa: F401 — registers the 'ti' namespace


@pytest.fixture(scope="module")
def df():
    n = 40
    return pl.DataFrame(
        {
            "open": [1.0 + i * 0.1 for i in range(n)],
            "high": [2.0 + i * 0.1 for i in range(n)],
            "low": [0.5 + i * 0.1 for i in range(n)],
            "close": [1.5 + i * 0.1 for i in range(n)],
            "volume": [100.0 + i for i in range(n)],
        }
    )


# (accessor method, {param: negative value}) — one row per formerly-crashing pair.
NEGATIVE_PERIOD_CASES = [
    # slice 1: trend / cycles
    ("adx", {"length": -10}),
    ("adx", {"lensig": -10}),
    ("adx", {"adxr_length": -10}),
    ("adxr", {"length": -10}),
    ("amat", {"fast": -10}),
    ("amat", {"slow": -10}),
    ("pmax", {"length": -10}),
    ("trendflex", {"length": -10}),
    ("trendflex", {"smooth": -10}),
    ("reflex", {"length": -10}),
    ("reflex", {"smooth": -10}),
    ("coppock", {"length": -10}),
    ("coppock", {"fast": -10}),
    ("coppock", {"slow": -10}),
    ("cg", {"length": -10}),
    # slice 2: momentum
    ("cfo", {"length": -10}),
    ("cmo", {"length": -10}),
    ("cmo", {"drift": -10}),
    ("inertia", {"length": -10}),
    ("inertia", {"rvi_length": -10}),
    ("po", {"length": -10}),
    ("rsx", {"length": -10}),
    ("stc", {"tclength": -10}),
    ("stc", {"fast": -10}),
    ("stc", {"slow": -10}),
    ("mom", {"length": -10}),
    ("exhc", {"length": -10}),
    # slice 3: overlap / volume
    ("dema", {"length": -10}),
    ("mmar", {"length": -10}),
    ("wma", {"length": -10}),
    ("zlma", {"length": -10}),
    ("aobv", {"fast": -10}),
    ("aobv", {"slow": -10}),
    ("efi", {"length": -10}),
    ("kvo", {"fast": -10}),
    ("kvo", {"slow": -10}),
    ("kvo", {"signal": -10}),
    ("pvi", {"length": -10}),
    # slice 4: volatility / statistics / volume
    ("kc", {"length": -10}),
    ("massi", {"fast": -10}),
    ("massi", {"slow": -10}),
    ("natr", {"length": -10}),
    ("thermo", {"length": -10}),
    ("mad", {"length": -10}),
    ("md", {"length": -10}),
    ("quantile", {"length": -10}),
    ("pvo", {"fast": -10}),
    ("pvo", {"slow": -10}),
    ("pvo", {"signal": -10}),
    # parent-hardened: root kernels + their delegators
    ("cvi", {"length": -10}),
    ("dx", {"length": -10}),
    ("ema", {"length": -10}),
    ("linreg", {"length": -10}),
    ("t3", {"length": -10}),
    ("tema", {"length": -10}),
    ("cti", {"length": -10}),  # delegates to linreg (validated transitively)
    ("dsp", {"length": -10}),  # delegates to ema
    ("tsf", {"length": -10}),  # delegates to linreg
    ("dm", {"length": -10}),
    ("dm", {"drift": -10}),
]


@pytest.mark.parametrize("method, kwargs", NEGATIVE_PERIOD_CASES, ids=lambda v: v if isinstance(v, str) else str(v))
def test_negative_period_raises_valueerror(df, method, kwargs):
    with pytest.raises(ValueError):
        getattr(df.ti, method)(**kwargs)


def test_v_pos_int_contract():
    """The shared validator: rejects non-int, bool, and < minimum; passes valid."""
    from polars_ti.utils import v_pos_int

    assert v_pos_int(14, "length") == 14
    assert v_pos_int(1, "length") == 1
    with pytest.raises(ValueError):
        v_pos_int(0, "length")
    with pytest.raises(ValueError):
        v_pos_int(-5, "length")
    with pytest.raises(ValueError):
        v_pos_int(3.5, "length")  # non-integer
    with pytest.raises(ValueError):
        v_pos_int(True, "length")  # bool rejected
    with pytest.raises(ValueError):
        v_pos_int(None, "length")


# Weight-based moving averages built an O(length) weight vector eagerly, so an
# absurd length (>> data) allocated gigabytes and hung/OOM'd. Weights are now
# built lazily inside the rolling closure — only once a full window exists — so a
# length larger than the data returns all-null (the mathematically correct result)
# without the allocation.
@pytest.mark.parametrize("indicator", ["alma", "swma", "sinwma", "fwma", "pwma"])
def test_weighted_ma_huge_length_returns_null_not_hang(indicator):
    df = pl.DataFrame({"close": [1.0 + i * 0.1 for i in range(40)]})
    result = df.select(getattr(ti, indicator)("close", length=10**9).alias("x"))
    col = result.get_column("x")
    # length >> n: no full window exists, so every value is null/NaN (no hang).
    assert col.is_null().all() or col.is_nan().all()


# cg and msw also eagerly allocated an O(length)/O(period) weight/basis array
# before a window loop that is empty when the window exceeds the data. Guarded to
# return all-null (cg) / all-NaN (msw) without the allocation.
def test_cg_huge_length_returns_null_not_hang():
    df = pl.DataFrame({"close": [1.0 + i * 0.1 for i in range(40)]})
    col = df.select(ti.cg("close", length=10**9).alias("x")).get_column("x")
    assert col.is_null().all() or col.is_nan().all()


def test_msw_huge_period_returns_nan_not_hang():
    df = pl.DataFrame({"close": [1.0 + i * 0.1 for i in range(40)]})
    out = df.select(ti.msw("close", period=10**9).alias("s")).unnest("s")
    assert out["sine"].is_nan().all() or out["sine"].is_null().all()


def test_msw_negative_period_raises():
    df = pl.DataFrame({"close": [1.0 + i * 0.1 for i in range(40)]})
    with pytest.raises(ValueError):
        df.ti.msw(period=-10)
