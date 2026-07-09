# -*- coding: utf-8 -*-
"""Tests for pl_alphatrend, including restored volume/MFI and src params."""

import numpy as np
import polars as pl
import pytest
from polars_ti.trend.alphatrend import alphatrend


class TestPlAlphatrend:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        np.random.seed(42)
        n = 120
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.1
        volume = np.abs(np.random.randn(n) * 1000) + 100
        return pl.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})

    def test_returns_expr(self):
        assert isinstance(alphatrend("high", "low", "close"), pl.Expr)

    def test_columns_present(self, sample_df):
        result = sample_df.select(alphatrend("high", "low", "close"))
        result = result.unnest(result.columns[0])
        assert "ALPHAT_14_1_50" in result.columns
        assert "ALPHATl_14_1_50_2" in result.columns

    def _trend(self, df, **kw):
        r = df.select(alphatrend("high", "low", "close", **kw))
        return r.unnest(r.columns[0])["ALPHAT_14_1_50"].to_numpy()

    def test_volume_mfi_and_src_restored(self, sample_df):
        """Regression: restored OLD volume->MFI momentum path and src selector.

        Passing the defaults explicitly (no volume, src='close') must reproduce
        the default output byte-for-byte, while the volume (MFI) path and a
        non-default src must change the result.
        """
        default = self._trend(sample_df)

        # Explicit default src reproduces the default exactly.
        passthrough = self._trend(sample_df, src="close")
        np.testing.assert_array_equal(np.isnan(default), np.isnan(passthrough))
        m = ~np.isnan(default)
        assert np.array_equal(default[m], passthrough[m])

        # Providing volume switches momentum to MFI and changes the output.
        with_volume = self._trend(sample_df, volume="volume")
        assert not np.allclose(default, with_volume, equal_nan=True)

        # A non-default RSI source changes the output.
        src_low = self._trend(sample_df, src="low")
        assert not np.allclose(default, src_low, equal_nan=True)

    def test_src_open_requires_open(self):
        """src='open' uses the supplied open series and changes the output."""
        np.random.seed(11)
        n = 120
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        # An 'open' series materially different from 'close' so the RSI source
        # selection is observable.
        open_ = close + np.cumsum(np.random.randn(n) * 0.4)
        df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})

        default = self._trend(df)
        src_open = self._trend(df, src="open", open_="open")
        assert not np.allclose(default, src_open, equal_nan=True)

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(alphatrend("high", "low", "close")).collect()
        assert result.height == 120
