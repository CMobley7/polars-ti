# -*- coding: utf-8 -*-
"""Tests for pl_eri - Polars + pl_ema implementation."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.eri import pl_eri


class TestPlEri:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        high = 101 + np.cumsum(np.random.randn(n) * 0.5)
        low = high - np.abs(np.random.randn(n) * 0.3)
        close = (high + low) / 2
        return pl.DataFrame({'high': high, 'low': low, 'close': close})

    def test_returns_list_of_expr(self):
        result = pl_eri("high", "low", "close")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(e, pl.Expr) for e in result)

    def test_has_eri_columns(self, sample_df):
        result = sample_df.select(pl_eri("high", "low", "close"))
        assert "BULLP_13" in result.columns
        assert "BEARP_13" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_eri("high", "low", "close"))
        assert result["BULLP_13"][20:].is_nan().sum() == 0
        assert result["BEARP_13"][20:].is_nan().sum() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_eri("high", "low", "close", offset=0))
        result_with_offset = sample_df.select(pl_eri("high", "low", "close", offset=5))
        assert result_with_offset["BULLP_13"].null_count() > result_no_offset["BULLP_13"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_eri("high", "low", "close")).collect()
        assert "BULLP_13" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(pl_eri("high", "low", "close", length=20))
        assert "BULLP_20" in result.columns
        assert "BEARP_20" in result.columns

    def test_bull_greater_than_bear(self, sample_df):
        """Bull power should generally be > Bear power (high > low)."""
        result = sample_df.select(pl_eri("high", "low", "close"))
        valid_bull = result["BULLP_13"].filter(~result["BULLP_13"].is_nan())
        valid_bear = result["BEARP_13"].filter(~result["BEARP_13"].is_nan())
        assert valid_bull.mean() > valid_bear.mean()
