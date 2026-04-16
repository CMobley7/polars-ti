# -*- coding: utf-8 -*-
"""Tests for pl_crsi - Polars + Numba implementation."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.crsi import pl_crsi


class TestPlCrsi:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200  # Need enough for length_rank=100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({'close': close})

    def test_returns_expr(self):
        expr = pl_crsi("close")
        assert isinstance(expr, pl.Expr)

    def test_has_crsi_column(self, sample_df):
        result = sample_df.select(pl_crsi("close"))
        assert "CRSI_3_2_100" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_crsi("close"))
        # After warmup (100 + some buffer), should have values
        assert result["CRSI_3_2_100"][120:].is_nan().sum() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_crsi("close", offset=0, talib=False))
        result_with_offset = sample_df.select(pl_crsi("close", offset=5, talib=False))
        assert result_with_offset["CRSI_3_2_100"].null_count() > result_no_offset["CRSI_3_2_100"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_crsi("close")).collect()
        assert "CRSI_3_2_100" in result.columns

    def test_talib_true(self, sample_df):
        result = sample_df.select(pl_crsi("close", talib=True))
        assert "CRSI_3_2_100" in result.columns

    def test_talib_false(self, sample_df):
        result = sample_df.select(pl_crsi("close", talib=False))
        assert "CRSI_3_2_100" in result.columns

    def test_custom_parameters(self, sample_df):
        result = sample_df.select(pl_crsi("close", length_rsi=5, length_streak=3, length_rank=50))
        assert "CRSI_5_3_50" in result.columns

    def test_values_in_range(self, sample_df):
        """CRSI should be between 0 and 100."""
        result = sample_df.select(pl_crsi("close"))
        valid = result["CRSI_3_2_100"].filter(~result["CRSI_3_2_100"].is_nan())
        assert valid.min() >= 0
        assert valid.max() <= 100
