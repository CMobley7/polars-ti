# -*- coding: utf-8 -*-
"""Tests for pl_percent_return."""
import numpy as np
import polars as pl
import pytest
from polars_ti.performance.percent_return import pl_percent_return


class TestPlPercentReturn:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        return pl.DataFrame({'close': 100 + np.cumsum(np.random.randn(100) * 0.5)})

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        return {
            'pd_close': close,
            'pl_df': pl.DataFrame({'close': close}),
        }

    def test_returns_expression(self):
        result = pl_percent_return("close")
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_df):
        result = sample_df.select(pl_percent_return("close", length=1))
        assert "PCTRET_1" in result.columns

    def test_cumulative_mode(self, sample_data):
        """Cumulative percent return works."""
        pl_result = sample_data['pl_df'].select(pl_percent_return('close', cumulative=True)).to_series()
        assert "CUMPCTRET" in pl_result.name

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(pl_percent_return("close"))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values (produces NaN/inf)."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(pl_percent_return("close"))
        assert result.height == 50

    def test_lazy_execution(self, sample_df):
        """Works with LazyFrame."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_percent_return("close")).collect()
        assert "PCTRET_1" in result.columns
