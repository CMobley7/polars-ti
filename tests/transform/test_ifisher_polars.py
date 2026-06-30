# -*- coding: utf-8 -*-
"""Tests for pl_ifisher."""

import numpy as np
import polars as pl
import pytest
from polars_ti.transform.ifisher import ifisher


class TestPlIfisher:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        close = np.random.randn(100) * 0.5
        return np.clip(close, -1, 1)

    def test_returns_expressions(self, sample_data):
        result = ifisher("close")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(e, pl.Expr) for e in result)

    def test_output_has_correct_alias(self, sample_data):
        df = pl.DataFrame({"close": sample_data})
        result = df.select(ifisher("close", amp=1.0))
        assert "INVFISHER_1.0" in result.columns
        assert "INVFISHERs_1.0" in result.columns

    def test_with_null_values(self, sample_data):
        data = sample_data.copy()
        data[10:15] = np.nan
        df = pl.DataFrame({"close": data})
        result = df.select(ifisher("close"))
        assert result.height == 100

    def test_with_zeros(self, sample_data):
        data = sample_data.copy()
        data[20:25] = 0.0
        df = pl.DataFrame({"close": data})
        result = df.select(ifisher("close"))
        assert result.height == 100

    def test_lazy_execution(self, sample_data):
        df = pl.DataFrame({"close": sample_data})
        result = df.lazy().select(ifisher("close")).collect()
        assert result.height == 100
