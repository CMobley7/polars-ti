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
    return pl.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


@pytest.fixture
def pd_ohlcv(ohlcv_df):
    """Pandas DataFrameversion of OHLCV data (no pyarrow needed)."""
    return pl.DataFrame({
        col: ohlcv_df[col].to_numpy() for col in ohlcv_df.columns
    })


class TestPlDpo:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.dpo import pl_dpo
        result = ohlcv_df.select(pl_dpo("close"))
        assert result.height == 200

class TestPlQstick:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.qstick import pl_qstick
        result = ohlcv_df.select(pl_qstick("open", "close"))
        assert result.height == 200

class TestPlVhf:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.vhf import pl_vhf
        result = ohlcv_df.select(pl_vhf("close"))
        assert result.height == 200

class TestPlDecay:
    def test_linear(self, ohlcv_df):
        from polars_ti.trend.decay import pl_decay
        result = ohlcv_df.select(pl_decay("close", length=5))
        assert result.height == 200

class TestPlTtmTrend:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.ttm_trend import pl_ttm_trend
        result = ohlcv_df.select(pl_ttm_trend("high", "low", "close"))
        assert result.height == 200


class TestPlTsignals:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.tsignals import pl_tsignals
        trend = pl.col("close") > pl.col("close").rolling_mean(window_size=20)
        result = ohlcv_df.select(pl_tsignals(trend))
        assert result.height == 200


class TestPlChop:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.chop import pl_chop
        result = ohlcv_df.select(pl_chop("high", "low", "close"))
        assert result.height == 200


class TestPlVortex:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.vortex import pl_vortex
        result = ohlcv_df.select(pl_vortex("high", "low", "close"))
        assert result.height == 200


class TestPlAroon:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.aroon import pl_aroon
        result = ohlcv_df.select(pl_aroon("high", "low"))
        assert result.height == 200


class TestPlAdx:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.adx import pl_adx
        result = ohlcv_df.select(pl_adx("high", "low", "close"))
        assert result.height == 200


class TestPlPsar:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.psar import pl_psar
        result = ohlcv_df.select(pl_psar("high", "low"))
        assert result.height == 200


class TestPlCksp:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.cksp import pl_cksp
        result = ohlcv_df.select(pl_cksp("high", "low", "close"))
        assert result.height == 200


class TestPlRwi:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.rwi import pl_rwi
        result = ohlcv_df.select(pl_rwi("high", "low", "close"))
        assert result.height == 200


class TestPlHtTrendline:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.ht_trendline import pl_ht_trendline
        result = ohlcv_df.select(pl_ht_trendline("close"))
        assert result.height == 200


class TestPlTrama:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.trama import pl_trama
        result = ohlcv_df.select(pl_trama("close"))
        assert result.height == 200


class TestPlTrendflex:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.trendflex import pl_trendflex
        result = ohlcv_df.select(pl_trendflex("close"))
        assert result.height == 200


class TestPlZigzag:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.zigzag import pl_zigzag
        result = ohlcv_df.select(pl_zigzag("high", "low"))
        assert result.height == 200


class TestPlPmax:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.pmax import pl_pmax
        result = ohlcv_df.select(pl_pmax("high", "low", "close"))
        assert result.height == 200


class TestPlAlphatrend:
    def test_returns_expression(self, ohlcv_df):
        from polars_ti.trend.alphatrend import pl_alphatrend
        result = ohlcv_df.select(pl_alphatrend("high", "low", "close"))
        assert result.height == 200
