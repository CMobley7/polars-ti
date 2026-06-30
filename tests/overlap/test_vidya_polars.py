# -*- coding: utf-8 -*-
"""Unit tests for polars_ti/overlap/vidya.py Polars implementation."""

import numpy as np
import polars as pl
import pytest

from polars_ti.overlap.vidya import vidya


class TestPlVidya:
    """Tests for pl_vidya - Variable Index Dynamic Average."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        close = 100 + np.random.randn(200).cumsum()
        return {
            "pd_series": close,
            "pl_df": pl.DataFrame({"close": close}),
        }

    def test_returns_expression(self):
        """Returns a Polars expression."""
        result = vidya("close", length=14)
        assert isinstance(result, pl.Expr)

    def test_output_has_correct_alias(self, sample_data):
        """Output column has correct alias."""
        result = sample_data["pl_df"].select(vidya("close", length=14))
        assert result.columns[0] == "VIDYA_14"

    def test_with_null_values(self):
        """Handles null values gracefully."""
        df = pl.DataFrame({"close": [None] + [100.0] * 49})
        result = df.select(vidya("close", length=14))
        assert result.height == 50

    def test_with_zeros(self):
        """Handles zero values."""
        df = pl.DataFrame({"close": [0.0] * 5 + [100.0] * 45})
        result = df.select(vidya("close", length=14))
        assert result.height == 50

    def test_lazy_execution(self, sample_data):
        """Works with LazyFrame."""
        lazy_df = sample_data["pl_df"].lazy()
        result = lazy_df.select(vidya("close", length=14)).collect()
        assert "VIDYA_14" in result.columns

    @staticmethod
    def _classic_vidya(close: np.ndarray, length: int = 14, drift: int = 1) -> np.ndarray:
        """Independent reproduction of pandas-ta-classic's vidya (commit 1474768).

        Inlined CMO (rolling pos/neg sums, NOT scaled by 100, no TA-Lib), SMA
        seed at index length-1, then the convex-combination recurrence."""
        n = close.size
        mom = np.concatenate([np.full(drift, np.nan), close[drift:] - close[:-drift]])
        pos = np.clip(mom, 0, None)
        neg = np.abs(np.clip(mom, None, 0))
        pos_sum = pl.Series(pos).rolling_sum(window_size=length, min_samples=length).to_numpy()
        neg_sum = pl.Series(neg).rolling_sum(window_size=length, min_samples=length).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            cmo = (pos_sum - neg_sum) / (pos_sum + neg_sum)
        abs_cmo = np.abs(cmo)
        alpha = 2.0 / (length + 1)
        out = np.full(n, np.nan)
        out[length - 1] = close[:length].mean()
        for i in range(length, n):
            out[i] = alpha * abs_cmo[i] * close[i] + out[i - 1] * (1 - alpha * abs_cmo[i])
        return out

    def test_native_matches_classic_fork(self):
        """classic 1474768: SMA-seeded VIDYA. There is no TA-Lib VIDYA, so the
        native (talib=False) output is validated against the classic fork's
        formula (inlined native CMO). The old zero-seed produced a materially
        wrong transient; the SMA seed matches the classic reference."""
        df = pl.read_csv("data/SPY_D.csv", try_parse_dates=True).head(1500)
        close = df["close"].to_numpy().astype(float)
        ref = self._classic_vidya(close, length=14, drift=1)
        got = df.select(vidya("close", length=14, drift=1, talib=False)).to_series().to_numpy()
        # Seed lands at index length-1 (13).
        assert not np.isnan(got[13])
        assert np.all(np.isnan(got[:13]))
        mask = ~np.isnan(ref) & ~np.isnan(got)
        assert mask.sum() > 1000
        assert np.max(np.abs(ref[mask] - got[mask])) < 1e-9
