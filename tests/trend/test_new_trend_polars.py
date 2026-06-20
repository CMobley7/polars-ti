# -*- coding: utf-8 -*-
"""Tests for all newly implemented trend Polars indicators."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def ohlcv_df() -> pl.DataFrame:
    """Generate reproducible OHLCV test data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n).astype(float)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def pd_ohlcv(ohlcv_df):
    """Pandas DataFrameversion of OHLCV data (no pyarrow needed)."""
    return pl.DataFrame({col: ohlcv_df[col].to_numpy() for col in ohlcv_df.columns})


class TestPlDpo:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.dpo import dpo

        result = ohlcv_df.select(dpo("close"))
        assert result.height == 200


class TestPlQstick:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.qstick import qstick

        result = ohlcv_df.select(qstick("open", "close"))
        assert result.height == 200


class TestPlVhf:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.vhf import vhf

        result = ohlcv_df.select(vhf("close"))
        assert result.height == 200


class TestPlDecay:
    def test_linear(self, ohlcv_df):
        from polars_ti.trend.decay import decay

        result = ohlcv_df.select(decay("close", length=5))
        assert result.height == 200


class TestPlTtmTrend:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.ttm_trend import ttm_trend

        result = ohlcv_df.select(ttm_trend("high", "low", "close"))
        assert result.height == 200


class TestPlTsignals:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.tsignals import tsignals

        trend = pl.col("close") > pl.col("close").rolling_mean(window_size=20)
        result = ohlcv_df.select(tsignals(trend))
        assert result.height == 200


class TestPlChop:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.chop import chop

        result = ohlcv_df.select(chop("high", "low", "close"))
        assert result.height == 200


class TestPlVortex:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.vortex import vortex

        result = ohlcv_df.select(vortex("high", "low", "close"))
        assert result.height == 200


class TestPlAroon:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.aroon import aroon

        result = ohlcv_df.select(aroon("high", "low"))
        assert result.height == 200


class TestPlAdx:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.adx import adx

        result = ohlcv_df.select(adx("high", "low", "close"))
        assert result.height == 200


class TestPlPsar:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.psar import psar

        result = ohlcv_df.select(psar("high", "low"))
        assert result.height == 200


class TestPlCksp:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.cksp import cksp

        result = ohlcv_df.select(cksp("high", "low", "close"))
        assert result.height == 200


class TestPlRwi:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.rwi import rwi

        result = ohlcv_df.select(rwi("high", "low", "close"))
        assert result.height == 200


class TestPlHtTrendline:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.ht_trendline import ht_trendline

        result = ohlcv_df.select(ht_trendline("close"))
        assert result.height == 200


class TestPlTrama:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.trama import trama

        result = ohlcv_df.select(trama("close"))
        assert result.height == 200


class TestPlTrendflex:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.trendflex import trendflex

        result = ohlcv_df.select(trendflex("close"))
        assert result.height == 200


class TestPlZigzag:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.zigzag import zigzag

        result = ohlcv_df.select(zigzag("high", "low"))
        assert result.height == 200


class TestPlPmax:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.pmax import pmax

        result = ohlcv_df.select(pmax("high", "low", "close"))
        assert result.height == 200


class TestPlAlphatrend:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.alphatrend import alphatrend

        result = ohlcv_df.select(alphatrend("high", "low", "close"))
        assert result.height == 200
