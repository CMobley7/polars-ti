# -*- coding: utf-8 -*-
"""Tests for pl_imi - Pure Polars implementation."""
import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.imi import pl_imi


class TestPlImi:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        open_ = close + np.random.randn(n) * 0.3
        return pl.DataFrame({'open': open_, 'close': close})

    def test_returns_expr(self):
        expr = pl_imi("open", "close")
        assert isinstance(expr, pl.Expr)

    def test_has_imi_column(self, sample_df):
        result = sample_df.select(pl_imi("open", "close"))
        assert "IMI_14" in result.columns

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_imi("open", "close"))
        assert result["IMI_14"][20:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result_no_offset = sample_df.select(pl_imi("open", "close", offset=0))
        result_with_offset = sample_df.select(pl_imi("open", "close", offset=5))
        assert result_with_offset["IMI_14"].null_count() > result_no_offset["IMI_14"].null_count()

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_imi("open", "close")).collect()
        assert "IMI_14" in result.columns

    def test_custom_length(self, sample_df):
        result = sample_df.select(pl_imi("open", "close", length=20))
        assert "IMI_20" in result.columns

    def test_values_in_range(self, sample_df):
        """IMI should be between 0 and 100."""
        result = sample_df.select(pl_imi("open", "close"))
        valid = result["IMI_14"].filter(~result["IMI_14"].is_null())
        assert valid.min() >= 0
        assert valid.max() <= 100
