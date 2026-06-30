# -*- coding: utf-8 -*-
import sys
from os.path import dirname

sys.dont_write_bytecode = True

# Make the reusable parity helpers (tests/_parity.py, tests/parity_exceptions.py)
# importable by their bare module names from any test module, regardless of
# pytest's package import mode.
_TESTS_DIR = dirname(__file__)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Legacy pandas-era root-level test files that cannot be collected safely.
# These retain the original pandas-based test logic and are excluded from
# the active Polars test suite until they are individually refactored.
collect_ignore = [
    "test_indicator_candle.py",
    "test_indicator_cycles.py",
    "test_indicator_momentum.py",
    "test_indicator_overlap.py",
    "test_indicator_performance.py",
    "test_indicator_statistics.py",
    "test_indicator_transform.py",
    "test_indicator_trend.py",
    "test_indicator_volatility.py",
    "test_indicator_volume.py",
    "test_metrics.py",
    "test_studies.py",
    "test_supertrend_verification.py",
    "test_utils.py",
]


from os import system as os_system

import pytest
import polars as pl

import polars_ti as ti
from polars_ti.maps import Imports

# Test hook: force the library into its no-TA-Lib (native) code paths even when
# TA-Lib is installed, so the no-TA-Lib behaviour can be exercised locally with
# ``POLARS_TI_SIMULATE_NO_TALIB=1 uv run pytest``. In CI the no-TA-Lib leg simply
# doesn't install TA-Lib, so this is only a convenience for local verification.
if __import__("os").environ.get("POLARS_TI_SIMULATE_NO_TALIB") == "1":
    Imports["talib"] = False
    # Make ``import talib`` raise ImportError too, so try/except-guarded code
    # paths behave exactly as in a real TA-Lib-absent environment.
    sys.modules["talib"] = None

# Whether TA-Lib is importable in this environment. Parity tests that grade
# against the TA-Lib golden (old_talib / talib_reference) must skip when absent,
# so the no-TA-Lib CI leg stays green (it exercises the native code paths).
HAS_TALIB = bool(Imports.get("talib", False))
requires_talib = pytest.mark.skipif(not HAS_TALIB, reason="requires TA-Lib")

# Pre-warm all Numba JIT kernels once per session to prevent concurrent
# JIT compilation crashes when multiple indicator modules are loaded together.
import polars_ti.momentum
import polars_ti.volatility
import polars_ti.overlap

TEST_ROWS = 200
TEST_CSV = f"data/SPY_D.csv"


BEEP = False
PLAY_BEEP = f"osascript -e beep"


@pytest.fixture(name="df", scope="function")
def testdf():
    """Yields a truncated Polars df from TEST_CSV file."""
    df = pl.read_csv(TEST_CSV, try_parse_dates=True)
    df = df.drop(["dividends", "stock splits"])
    yield df.head(TEST_ROWS)

    if BEEP:
        os_system(PLAY_BEEP)


@pytest.fixture(scope="function")
def all_study():
    """Returns the All Study"""
    return ti.AllStudy


@pytest.fixture(scope="function")
def common_study():
    """Returns the Common Study"""
    return ti.CommonStudy


@pytest.fixture(scope="function")
def custom_study_a():
    """Returns a Custom Study with a chained/composed indicator: 'ema(CUMLOGRET_1, 5)'.
    This Study only works when cores=0. When using mulptiprocessing (cores > 0),
    the multiprocesser might miss the results of the indicator
    'CUMLOGRET_1 = log_return(cumulative=True)'
    """
    _ti = [
        {"kind": "cdl_pattern", "name": "tristar"},  # 1
        {"kind": "rsi"},  # 1
        {"kind": "macd"},  # 3
        {"kind": "sma", "length": 50},  # 1
        {"kind": "trix"},  # 2
        {"kind": "bbands", "length": 20},  # 5
        {"kind": "log_return", "cumulative": True},  # 1
        {"kind": "ema", "close": "CUMLOGRET_1", "length": 5, "suffix": "CLR"},  # 1
    ]
    return ti.Study(
        name="Commons with Cumulative Log Return EMA Chain",
        ti=_ti,
        # cores=0,
        description="Common indicators with specific lengths and a chained indicator",
    )


@pytest.fixture(scope="function")
def custom_study_b():
    """Returns a Custom Study that allows setting indicator values by
    parameter index as a tuple instead of using a named parameter"""
    _ti = [
        {"kind": "ema", "params": (5,)},  # 1
        {"kind": "fisher", "params": (13, 7)},  # 2
    ]
    return ti.Study(
        name="Custom Args Tuple",
        ti=_ti,
        description="Allow for easy filling in indicator arguments by argument placement",
    )


@pytest.fixture(scope="function")
def custom_study_c():
    """Returns a Custom Study that makes it easy to rename individual
    indicator resultant column names"""
    return ti.Study(
        name="Custom Col Numbers Tuple",
        ti=[{"kind": "bbands", "col_names": ("LB", "MB", "UB", "BW", "BP")}],
        description="Allow for easy renaming of resultant columns",
    )


@pytest.fixture(scope="function")
def custom_study_d():
    """Returns a Custom Study that makes it easily return individual
    indicator resultant columns by column number (col_numbers) as a tuple"""
    return ti.Study(
        name="Custom Col Numbers Tuple",
        ti=[
            {"kind": "macd", "col_numbers": (1,)},  # macd histogram
            {"kind": "bbands", "col_numbers": (0, 2)},  # bbands lower and upper
        ],
        description="Allow for easy selection of resultant columns",
    )


@pytest.fixture(scope="function")
def custom_study_e():
    """Returns a Custom Study that has non default indicator parameters and
    an example of indicator composition/chaining: 'ema(CUMLOGRET_1, 5)'"""
    _ti = [
        {"kind": "amat", "fast": 20, "slow": 50},  # 2
        {"kind": "log_return", "cumulative": True},  # 1
        {"kind": "ema", "close": "CUMLOGRET_1", "length": 5},  # 1
    ]

    return ti.Study(name="AMAT Log Returns", ti=_ti, cores=0, description="AMAT Log Returns")
