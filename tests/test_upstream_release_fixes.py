"""Regression checks for applicable pandas-ta-classic 0.8.32 release fixes."""

import inspect

import numpy as np
import polars as pl
import pytest

import polars_ti as ti


PREFIX_INDICATORS = [
    "adosc",
    "fisher",
    "hwc",
    "hwma",
    "jma",
    "kama",
    "lrsi",
    "macd",
    "mama",
    "mcgd",
    "ssf",
    "tos_stdevall",
    "vidya",
]


def prices(rows: int = 240) -> pl.DataFrame:
    """Return deterministic OHLCV input with enough nontrivial candle patterns."""
    rng = np.random.default_rng(1832)
    close = 100 + np.cumsum(rng.normal(size=rows))
    opening = close + rng.normal(scale=0.8, size=rows)
    return pl.DataFrame(
        {
            "open": opening,
            "high": np.maximum(opening, close) + rng.uniform(0.1, 1.0, rows),
            "low": np.minimum(opening, close) - rng.uniform(0.1, 1.0, rows),
            "close": close,
            "volume": rng.uniform(100, 1000, rows),
        }
    )


def values(frame: pl.DataFrame) -> np.ndarray:
    """Flatten a single public struct result while preserving row positions."""
    if isinstance(frame.dtypes[0], pl.Struct):
        frame = frame.unnest(frame.columns[0])
    return frame.to_numpy().astype(float)


@pytest.mark.parametrize("name", PREFIX_INDICATORS)
@pytest.mark.parametrize("offset", [0, 2, -2])
def test_recursive_indicators_skip_only_leading_missing_rows(name, offset):
    frame = prices()
    padded = pl.concat([pl.DataFrame({column: [float("nan")] * 17 for column in frame.columns}), frame])
    function = getattr(ti, name)
    arguments = {"offset": 0}
    if "talib" in inspect.signature(function).parameters:
        arguments["talib"] = False
    if name == "adosc":
        expression = function("high", "low", "close", "volume", **arguments)
    elif name == "fisher":
        expression = function("high", "low", **arguments)
    else:
        expression = function("close", **arguments)
    expected = values(frame.select(expression))
    expected = np.vstack([np.full((17, expected.shape[1]), np.nan), expected])
    if offset:
        expected = np.roll(expected, offset, axis=0)
        expected[:offset] = np.nan if offset > 0 else expected[:offset]
        if offset < 0:
            expected[offset:] = np.nan
    arguments["offset"] = offset
    if name == "adosc":
        expression = function("high", "low", "close", "volume", **arguments)
    elif name == "fisher":
        expression = function("high", "low", **arguments)
    else:
        expression = function("close", **arguments)
    actual = values(padded.select(expression))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True)


@pytest.mark.parametrize("name", ["sma", "ema", "wma", "dema", "tema", "trima", "kama", "mama", "t3"])
def test_ma_type_mapping_does_not_require_talib(name, monkeypatch):
    from polars_ti.maps import Imports
    from polars_ti.utils import tal_ma

    monkeypatch.setitem(Imports, "talib", False)
    assert tal_ma(name) == ["sma", "ema", "wma", "dema", "tema", "trima", "kama", "mama", "t3"].index(name)


@pytest.mark.parametrize("name", ["unknown", "", "linreg"])
def test_ma_type_mapping_rejects_unsupported_names(name):
    with pytest.raises(ValueError):
        ti.utils.tal_ma(name)


def test_ma_type_mapping_rejects_nonstring():
    with pytest.raises(TypeError):
        ti.utils.tal_ma(3)


@pytest.mark.parametrize("drift", [0, 2, 3])
def test_trade_signals_compare_consecutive_states(drift):
    frame = pl.DataFrame({"trend": [0, 1, 1, 1, 0, 0, 1]})
    expected = frame.select(ti.tsignals("trend"))
    with pytest.warns(DeprecationWarning, match="drift"):
        actual = frame.select(ti.tsignals("trend", drift=drift))
    assert actual.equals(expected)


CANDLE_PATTERNS = sorted(
    name
    for name in dir(ti)
    if name.startswith("cdl_")
    and "talib" in inspect.signature(getattr(ti, name)).parameters
    and name not in ("cdl_pattern",)
)


@pytest.mark.parametrize("name", CANDLE_PATTERNS)
def test_candle_prefix_matches_trimmed_input(name):
    frame = prices(600)
    padded = pl.concat([pl.DataFrame({column: [float("nan")] * 17 for column in frame.columns}), frame])
    expr = getattr(ti, name)("open", "high", "low", "close", talib=False)
    expected = values(frame.select(expr))
    actual = values(padded.select(expr))
    np.testing.assert_array_equal(actual[17:], expected)
    assert not actual[:17].any()


HT_NAMES = ["ht_dcperiod", "ht_dcphase", "ht_phasor", "ht_sine", "ht_trendmode", "ht_trendline"]


@pytest.mark.parametrize("name", HT_NAMES)
@pytest.mark.parametrize("use_talib", [False, True])
def test_hilbert_missing_prefix_and_gap(name, use_talib):
    if use_talib:
        from polars_ti.maps import Imports

        if not Imports["talib"]:
            pytest.skip("TA-Lib unavailable")
    frame = prices()
    padded = pl.concat([pl.DataFrame({column: [float("nan")] * 17 for column in frame.columns}), frame])
    expr = getattr(ti, name)("close", talib=use_talib)
    baseline = values(frame.select(expr))
    actual = values(padded.select(expr))
    np.testing.assert_allclose(actual[17:], baseline, rtol=2e-12, atol=2e-12, equal_nan=True)
    assert np.isnan(actual[:17]).all()
    gapped = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 150).then(float("nan")).otherwise(pl.col("close")).alias("close")
    )
    result = values(gapped.select(expr))
    np.testing.assert_allclose(result[:150], baseline[:150], rtol=2e-12, atol=2e-12, equal_nan=True)
    assert np.isnan(result[150:]).all()


@pytest.mark.parametrize("length", [2, 5, 14])
@pytest.mark.parametrize("prefix", [0, 17])
def test_native_adx_matches_talib_initialization(length, prefix):
    from polars_ti.maps import Imports

    if not Imports["talib"]:
        pytest.skip("TA-Lib unavailable")
    frame = prices()
    if prefix:
        frame = pl.concat([pl.DataFrame({column: [float("nan")] * prefix for column in frame.columns}), frame])
    native = values(frame.select(ti.adx("high", "low", "close", length=length, talib=False)))
    reference = values(frame.select(ti.adx("high", "low", "close", length=length, talib=True)))
    np.testing.assert_allclose(native, reference, rtol=2e-12, atol=2e-12, equal_nan=True)


def test_candle_full_normalization_is_causal_and_matches_expanding_oracle():
    import math

    frame = prices(80)
    exprs = ti.cdl_z("open", "high", "low", "close", full=True)
    actual = frame.select(exprs).to_numpy()
    np.testing.assert_allclose(frame.head(40).select(exprs).to_numpy(), actual[:40], rtol=0, atol=0, equal_nan=True)
    expected = np.full_like(actual, np.nan)
    for column, name in enumerate(["open", "high", "low", "close"]):
        data = frame[name].to_numpy()
        for index in range(1, len(data)):
            window = data[: index + 1]
            mean = math.fsum(window) / len(window)
            std = math.sqrt(math.fsum((float(value) - mean) ** 2 for value in window) / (len(window) - 1))
            expected[index, column] = (data[index] - mean) / std
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True)


def test_swma_length_one_is_identity():
    frame = pl.DataFrame({"close": [1.0, float("nan"), 3.0, 4.0]})
    np.testing.assert_array_equal(values(frame.select(ti.swma("close", length=1)))[:, 0], frame["close"].to_numpy())


def test_presma_rma_requires_complete_seed_window():
    frame = pl.DataFrame({"close": [float("nan"), 2.0, 3.0]})
    assert np.isnan(values(frame.select(ti.rma("close", length=3, presma=True)))).all()


@pytest.mark.parametrize("name", ["tos_stdevall", "vp"])
def test_full_sample_indicators_reject_causal_request(name):
    with pytest.raises(ValueError, match="lookahead"):
        if name == "vp":
            ti.vp(prices(), lookahead=False)
        else:
            ti.tos_stdevall("close", lookahead=False)


@pytest.mark.parametrize("with_sklearn", [False, True])
def test_linear_regression_is_fractional_and_reports_pearson_r(with_sklearn, monkeypatch):
    from polars_ti.maps import Imports

    monkeypatch.setitem(Imports, "sklearn", with_sklearn)
    x = pl.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = pl.Series([3.0, 2.5, 1.8, 1.7, 0.9])
    result = ti.utils.linear_regression(x, y)
    assert result["b"] == pytest.approx(-0.5)
    assert result["a"] == pytest.approx(1.98)
    assert result["r"] == pytest.approx(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1])
    np.testing.assert_allclose(result["line"], 1.98 - 0.5 * x.to_numpy())


@pytest.mark.parametrize("n,r,expected", [(3, 4, 0), (0, 0, 1), (5, 2, 10), (10**20, 1, 10**20)])
def test_combination_matches_integer_combinatorics(n, r, expected):
    assert ti.utils.combination(n, r) == expected


@pytest.mark.parametrize("length", [1, 2, 14])
@pytest.mark.parametrize("prefix", [0, 17])
def test_dm_matches_talib_with_prefix(length, prefix):
    from polars_ti.maps import Imports

    if not Imports["talib"]:
        pytest.skip("TA-Lib unavailable")
    frame = prices()
    if prefix:
        frame = pl.concat([pl.DataFrame({column: [float("nan")] * prefix for column in frame.columns}), frame])
    native = frame.select(ti.dm("high", "low", length=length, talib=False)).to_numpy()
    expected = frame.select(ti.dm("high", "low", length=length, talib=True)).to_numpy()
    np.testing.assert_allclose(native, expected, rtol=2e-12, atol=2e-12, equal_nan=True)


@pytest.mark.parametrize("name", PREFIX_INDICATORS + HT_NAMES)
@pytest.mark.parametrize("missing", [None, float("nan")])
def test_recursive_all_missing_is_undefined(name, missing):
    frame = pl.DataFrame({column: pl.Series([missing] * 20, dtype=pl.Float64) for column in prices().columns})
    function = getattr(ti, name)
    arguments = {"talib": False} if "talib" in inspect.signature(function).parameters else {}
    if name == "adosc":
        expr = function("high", "low", "close", "volume", **arguments)
    elif name == "fisher":
        expr = function("high", "low", **arguments)
    else:
        expr = function("close", **arguments)
    result = values(frame.select(expr))
    assert len(result) == len(frame)
    assert np.isnan(result).all()


def test_prefix_adapter_preserves_interior_missing_and_derived_warmup():
    from polars_ti.utils._prefix import run_after_prefix

    primary = np.array([np.nan, 1.0, 2.0, np.nan, 4.0])
    derived = np.array([np.nan, np.nan, 3.0, np.nan, 5.0])
    actual = run_after_prefix((primary, derived), lambda inputs: (inputs[1],), valid_inputs=1)[0]
    np.testing.assert_array_equal(actual, derived)


@pytest.mark.parametrize("arrays", [(), (np.array(1.0),), (np.zeros(2), np.zeros(3)), (np.zeros((2, 2)),)])
def test_prefix_adapter_rejects_invalid_shapes(arrays):
    from polars_ti.utils._prefix import run_after_prefix

    with pytest.raises(ValueError):
        run_after_prefix(arrays, lambda inputs: inputs)


@pytest.mark.parametrize("name", ["mama", "ht_trendline"])
@pytest.mark.parametrize("use_talib", [False, True])
def test_prenan_counts_from_first_valid_price(name, use_talib):
    from polars_ti.maps import Imports

    if use_talib and not Imports["talib"]:
        pytest.skip("TA-Lib unavailable")
    frame = prices()
    padded = pl.concat([pl.DataFrame({column: [float("nan")] * 17 for column in frame.columns}), frame])
    expr = getattr(ti, name)("close", prenan=80, talib=use_talib)
    actual = values(padded.select(expr))
    expected = values(frame.select(expr))
    np.testing.assert_array_equal(actual[17:], expected)
    assert np.isnan(actual[:17]).all()


@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_expanding_zscore_preserves_gaps_and_high_offset_precision(ddof):
    from polars_ti.candles.cdl_z import _expanding_zscore
    from decimal import Decimal, localcontext

    data = np.array([np.nan, 1e12, 1e12 + 0.001, np.nan, 1e12 - 0.002, 1e12 + 0.003])
    expected = np.full(len(data), np.nan)
    with localcontext() as context:
        context.prec = 60
        window = []
        for index, value in enumerate(data):
            if not np.isfinite(value):
                continue
            window.append(Decimal(float(value)))
            mean = sum(window) / len(window)
            moment = sum((item - mean) ** 2 for item in window)
            if len(window) > ddof and moment > 0:
                expected[index] = float((window[-1] - mean) / (moment / (len(window) - ddof)).sqrt())
    np.testing.assert_allclose(_expanding_zscore(data, ddof), expected, rtol=2e-12, atol=2e-12, equal_nan=True)


@pytest.mark.parametrize(
    "x,y", [([1.0], [2.0]), ([1.0, 2.0], [2.0]), ([1.0, np.nan], [2.0, 3.0]), ([1.0, 1.0], [2.0, 3.0])]
)
def test_regression_rejects_invalid_input(x, y):
    with pytest.raises(ValueError):
        ti.utils.linear_regression(np.array(x), np.array(y))


@pytest.mark.parametrize("prefix", [0, 17])
def test_true_range_requires_previous_close(prefix):
    frame = prices()
    if prefix:
        frame = pl.concat([pl.DataFrame({column: [float("nan")] * prefix for column in frame.columns}), frame])
    expected = np.full(len(frame), np.nan)
    high, low, close = (frame[name].to_numpy() for name in ("high", "low", "close"))
    for index in range(prefix + 1, len(frame)):
        expected[index] = max(
            high[index] - low[index], abs(high[index] - close[index - 1]), abs(low[index] - close[index - 1])
        )
    actual = values(frame.select(ti.true_range("high", "low", "close", talib=False)))[:, 0]
    np.testing.assert_array_equal(actual, expected)


def test_candle_color_does_not_classify_missing_prices():
    frame = pl.DataFrame({"open": [1.0, np.nan, 2.0, None], "close": [2.0, 2.0, np.nan, 1.0]})
    actual = frame.select(ti.utils.candle_color("open", "close")).to_series()
    assert actual.to_list() == [1, None, None, None]


@pytest.mark.parametrize("rows", [0, 1, 3])
@pytest.mark.parametrize("use_talib", [False, True])
def test_short_rvi_is_undefined(rows, use_talib):
    frame = prices(rows)
    actual = values(frame.select(ti.rvi("close", length=1, talib=use_talib)))
    assert actual.shape == (rows, 1)
    assert np.isnan(actual).all()


@pytest.mark.parametrize("name", ["pvt", "tos_stdevall"])
def test_short_summary_keeps_row_count(name):
    frame = prices(1)
    expr = ti.pvt("close", "volume", drift=10) if name == "pvt" else ti.tos_stdevall("close")
    actual = values(frame.select(expr))
    assert actual.shape[0] == 1
    assert np.isnan(actual).all()


@pytest.mark.parametrize("name", PREFIX_INDICATORS + HT_NAMES)
@pytest.mark.parametrize("rows", [0, 1, 3])
def test_recursive_short_input_preserves_shape(name, rows):
    frame = prices(rows)
    function = getattr(ti, name)
    arguments = {"talib": False} if "talib" in inspect.signature(function).parameters else {}
    if name == "adosc":
        expr = function("high", "low", "close", "volume", **arguments)
    elif name == "fisher":
        expr = function("high", "low", **arguments)
    else:
        expr = function("close", **arguments)
    assert frame.select(expr).height == rows


@pytest.mark.parametrize("name", ["dx", "adxr"])
@pytest.mark.parametrize("prefix", [0, 17])
@pytest.mark.parametrize("constant", [False, True])
def test_wilder_family_matches_talib_from_first_output(name, prefix, constant):
    from polars_ti.maps import Imports

    if not Imports["talib"]:
        pytest.skip("TA-Lib unavailable")
    frame = prices()
    if constant:
        frame = frame.with_columns(
            pl.lit(100.0).alias("high"), pl.lit(100.0).alias("low"), pl.lit(100.0).alias("close")
        )
    if prefix:
        frame = pl.concat([pl.DataFrame({column: [float("nan")] * prefix for column in frame.columns}), frame])
    function = getattr(ti, name)
    expected = values(frame.select(function("high", "low", "close", talib=True)))
    actual = values(frame.select(function("high", "low", "close", talib=False)))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True)


@pytest.mark.parametrize("missing_column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("asint", [False, True])
def test_doji_lookback_starts_after_joint_ohlc_prefix(missing_column, asint):
    frame = pl.DataFrame({"open": [1.0] * 37, "high": [2.0] * 37, "low": [0.0] * 37, "close": [1.0] * 37})
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) < 17).then(None).otherwise(pl.col(missing_column)).alias(missing_column)
    )
    expr = ti.cdl_doji("open", "high", "low", "close", talib=False, asint=asint)
    actual = frame.select(expr).to_series().to_numpy()
    expected = frame.slice(17).select(expr).to_series().to_numpy()
    np.testing.assert_array_equal(actual[17:], expected)
    assert not actual[:17].any()


@pytest.mark.parametrize("name", ["adx", "adxr", "dx"])
@pytest.mark.parametrize("scalar", [-100.0, 0.0, 30.0])
def test_wilder_scalar_is_applied_once(name, scalar):
    from polars_ti.maps import Imports

    if not Imports["talib"]:
        pytest.skip("TA-Lib unavailable")
    frame = prices()
    function = getattr(ti, name)
    actual = values(frame.select(function("high", "low", "close", scalar=scalar, talib=False)))
    expected = values(frame.select(function("high", "low", "close", scalar=scalar, talib=True)))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True)


def test_constant_response_regression_has_undefined_correlation_statistic():
    result = ti.utils.linear_regression(np.arange(5.0), np.ones(5))
    assert result["a"] == 1.0
    assert result["b"] == 0.0
    assert np.isnan(result["r"])
    assert np.isnan(result["t"])
    np.testing.assert_array_equal(result["line"], np.ones(5))


def test_nonpositive_trendline_prenan_adds_no_mask():
    frame = prices()
    expected = values(frame.select(ti.ht_trendline("close", prenan=0, talib=False)))
    actual = values(frame.select(ti.ht_trendline("close", prenan=-1, talib=False)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("use_talib", [False, True])
def test_kvo_flat_candles_have_zero_defined_outputs(use_talib):
    frame = pl.DataFrame({name: [100.0] * 120 for name in ["high", "low", "close", "volume"]})
    actual = frame.select(ti.kvo("high", "low", "close", "volume", talib=use_talib)).to_numpy()
    assert np.isfinite(actual[70:]).all()
    np.testing.assert_array_equal(actual[70:], 0.0)


@pytest.mark.parametrize("candle_range", [0.0, 1e-8])
def test_kvo_range_does_not_perturb_signed_volume_formula(candle_range):
    import math

    close = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 2.0, 2.0, 4.0, 5.0, 3.0])
    volume = np.arange(1.0, len(close) + 1) * 100
    frame = pl.DataFrame(
        {"high": close + candle_range / 2, "low": close - candle_range / 2, "close": close, "volume": volume}
    )
    force = np.concatenate(([np.nan], np.sign(np.diff(close)) * volume[1:]))

    def mean(values, length):
        result = np.full(len(values), np.nan)
        for index in range(length - 1, len(values)):
            window = values[index - length + 1 : index + 1]
            if np.isfinite(window).all():
                result[index] = math.fsum(window) / length
        return result

    oscillator = mean(force, 2) - mean(force, 3)
    expected = np.column_stack((oscillator, mean(oscillator, 2)))
    expr = ti.kvo("high", "low", "close", "volume", fast=2, slow=3, signal=2, mamode="sma", talib=False)
    actual = frame.select(expr).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True)
    np.testing.assert_array_equal(frame.head(8).select(expr).to_numpy(), actual[:8])
