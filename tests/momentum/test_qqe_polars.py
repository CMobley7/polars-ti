# -*- coding: utf-8 -*-
"""Tests for pl_qqe (Quantitative Qualitative Estimation)."""
import numpy as np
import polars as pl
import pytest
import pandas as pd  # REMOVED: pandas dependency  # Restored for fixtures
from polars_ti.momentum.qqe import pl_qqe


class TestPlQqe:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({'close': close})

    def test_returns_expression(self, sample_df):
        """Test that pl_qqe returns a valid expression."""
        result = sample_df.select(pl_qqe("close"))
        assert result.height == 200

    def test_output_struct_columns(self, sample_df):
        """Test that output struct has expected columns."""
        result = sample_df.select(pl_qqe("close"))
        unnested = result.unnest(result.columns[0])
        columns = unnested.columns
        assert any("QQE" in c for c in columns)
        assert any("QQEl" in c for c in columns)
        assert any("QQEs" in c for c in columns)

    def test_numerical_parity_pandas(self):
        """Numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        sample_df = pl.DataFrame({'close': close})
        sample_pdf = pd.DataFrame({'close': close})
        
        # Polars result
        polars_result = sample_df.select(pl_qqe("close", length=14, smooth=5))
        unnested = polars_result.unnest(polars_result.columns[0])
        polars_qqe = unnested.to_numpy()[:, 0]
        
        # Pandas result
        pandas_result = qqe(sample_pdf['close'], length=14, smooth=5)
        pandas_qqe = pandas_result.iloc[:, 0].to_numpy()
        
        # Compare after warmup
        warmup = 60
        mask = ~np.isnan(polars_qqe[warmup:]) & ~np.isnan(pandas_qqe[warmup:])
        max_diff = np.max(np.abs(polars_qqe[warmup:][mask] - pandas_qqe[warmup:][mask]))
        
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance"

    def test_with_null_values(self):
        """Test handling of null values."""
        df = pl.DataFrame({
            "close": [None] + [100.0] * 99
        })
        result = df.select(pl_qqe("close", length=14, smooth=5))
        assert result.height == 100

    def test_lazy_execution(self, sample_df):
        """Test that pl_qqe works with lazy frames."""
        lazy_df = sample_df.lazy()
        result = lazy_df.select(pl_qqe("close")).collect()
        assert result.height == 200

    def test_different_parameters(self, sample_df):
        """Test with different parameter values."""
        result = sample_df.select(pl_qqe("close", length=10, smooth=3, factor=3.0))
        unnested = result.unnest(result.columns[0])
        qqe_vals = unnested.to_numpy()[:, 0]
        assert (~np.isnan(qqe_vals)).sum() > 50
