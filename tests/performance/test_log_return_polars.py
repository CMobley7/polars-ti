# -*- coding: utf-8 -*-
"""Tests for pl_log_return."""

import numpy as np
import polars as pl
import pytest
from polars_ti.performance.log_return import pl_log_return


class TestPlLogReturn:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pl.DataFrame({"close": np.abs(close)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = np.abs(100 + np.random.randn(100).cumsum())
        return {
            "pd_close": close,
            "pl_df": pl.DataFrame({"close": close}),
        }

    def test_returns_expression(self):
        result = pl_log_return("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_log_return("close", length=1))
        assert "LOGRET_1" in result.columns

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_log_return("close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values (produces NaN/inf)."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(pl_log_return("close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_log_return("close")).collect()
        assert "LOGRET_1" in result.columns
