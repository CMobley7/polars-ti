# -*- coding: utf-8 -*-
"""Tests for pl_hwc."""

import numpy as np
import polars as pl
import pytest
from polars_ti.volatility.hwc import hwc


class TestPlHwc:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return {"close": close}

    def test_returns_expression(self, sample_data):
        result = hwc("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.select(hwc("close"))
        assert "HWC" in result.columns[0]

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][10:15] = np.nan
        df = pl.DataFrame(data)
        result = df.select(hwc("close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data["close"] = data["close"].copy()
        data["close"][50:55] = 50.0
        df = pl.DataFrame(data)
        result = df.select(hwc("close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame(sample_data)
        result = df.lazy().select(hwc("close")).collect()
        assert result.height == 100


class TestPlHwcChannels:
    """Regression: hwc(channels=True) must add HWW (width) and HWPCT
    (close's % position in the channel) matching the baseline formula, while
    the default (channels=False) output stays unchanged."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return {"close": close}

    def test_default_has_three_fields(self, sample_data):
        df = pl.DataFrame(sample_data)
        out = df.select(hwc("close")).to_series()
        assert list(out.struct.fields) == ["hwm", "hwu", "hwl"]

    def test_channels_adds_width_and_pct(self, sample_data):
        df = pl.DataFrame(sample_data)
        out = df.select(hwc("close", channels=True)).to_series()
        assert list(out.struct.fields) == ["hwm", "hwu", "hwl", "hww", "hwpct"]

    def test_channels_formula(self, sample_data):
        from sys import float_info as sflt

        df = pl.DataFrame(sample_data)
        out = df.select(hwc("close", channels=True)).unnest("HWC_1")
        upper = out["hwu"].to_numpy()
        lower = out["hwl"].to_numpy()
        width = out["hww"].to_numpy()
        pct = out["hwpct"].to_numpy()
        close = np.asarray(sample_data["close"], dtype=float)
        np.testing.assert_allclose(width, upper - lower)
        np.testing.assert_allclose(pct, (close - lower) / (width + sflt.epsilon))

    def test_channels_false_matches_default(self, sample_data):
        df = pl.DataFrame(sample_data)
        a = df.select(hwc("close"))
        b = df.select(hwc("close", channels=False))
        assert a.equals(b)

    def test_accessor_forwards_channels(self, sample_data):
        df = pl.DataFrame(sample_data)
        out = df.ti.hwc(channels=True).to_series()
        assert "hww" in out.struct.fields and "hwpct" in out.struct.fields
