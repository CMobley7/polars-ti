# -*- coding: utf-8 -*-
"""Tests for pl_fisher - Polars + Numba implementation."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.fisher import fisher as fisher_indicator


class TestPlFisher:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        high = 101 + np.cumsum(np.random.randn(n) * 0.5)
        low = high - np.abs(np.random.randn(n) * 0.3)
        return pl.DataFrame({"high": high, "low": low})

    def test_returns_list_of_expr(self):
        result = fisher_indicator("high", "low")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(e, pl.Expr) for e in result)

    def test_has_fisher_columns(self, sample_df):
        result = sample_df.select(fisher_indicator("high", "low"))
        assert "FISHERT_9_1" in result.columns
        assert "FISHERTs_9_1" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(fisher_indicator("high", "low"))
        assert result["FISHERT_9_1"][15:].is_nan().sum() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(fisher_indicator("high", "low", offset=0))
        result_with_offset = sample_df.select(fisher_indicator("high", "low", offset=5))
        assert result_with_offset["FISHERT_9_1"].null_count() > result_no_offset["FISHERT_9_1"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(fisher_indicator("high", "low")).collect()
        assert "FISHERT_9_1" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(fisher_indicator("high", "low", length=14, signal=3))
        assert "FISHERT_14_3" in result.columns
        assert "FISHERTs_14_3" in result.columns

    def test_signal_is_shifted_fisher(self, sample_df):
        """Signal should be Fisher shifted by 1."""
        result = sample_df.select(fisher_indicator("high", "low"))
        fisher = result["FISHERT_9_1"][:-1].to_numpy()
        signal = result["FISHERTs_9_1"][1:].to_numpy()
        valid = ~(np.isnan(fisher) | np.isnan(signal))
        assert np.allclose(fisher[valid], signal[valid])
