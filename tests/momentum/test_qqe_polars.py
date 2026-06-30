# -*- coding: utf-8 -*-
"""Tests for pl_qqe (Quantitative Qualitative Estimation)."""

import numpy as np
import polars as pl
import pytest
from polars_ti.momentum.qqe import qqe


class TestPlQqe:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"close": close})

    def test_returns_expression(self, sample_df):
        """Test that pl_qqe returns a valid expression."""
        result = sample_df.select(qqe("close"))
        assert result.height == 200

    def test_output_struct_columns(self, sample_df):
        """Test that output struct has expected columns."""
        result = sample_df.select(qqe("close"))
        unnested = result.unnest(result.columns[0])
        columns = unnested.columns
        assert any("QQE" in c for c in columns)
        assert any("QQEl" in c for c in columns)
        assert any("QQEs" in c for c in columns)

    def test_with_null_values(self):
        """Test handling of null values."""
        df = pl.DataFrame({"close": [None] + [100.0] * 99})
        result = df.select(qqe("close", length=14, smooth=5))
        assert result.height == 100

    def test_lazy_execution(self, sample_df):
        """Test that pl_qqe works with lazy frames."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(qqe("close")).collect()
        assert result.height == 200

    def test_different_parameters(self, sample_df):
        """Test with different parameter values."""
        result = sample_df.select(qqe("close", length=10, smooth=3, factor=3.0))
        unnested = result.unnest(result.columns[0])
        qqe_vals = unnested.to_numpy()[:, 0]
        assert (~np.isnan(qqe_vals)).sum() > 50
