# -*- coding: utf-8 -*-
"""Tests for pl_tmo - Polars + Numba True Momentum Oscillator."""
import numpy as np
import polars as pl
import pandas as pd
import pytest
from polars_ti.momentum.tmo import pl_tmo


class TestPlTmo:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        open_ = close - np.random.randn(n) * 0.3
        return pl.DataFrame({'open': open_, 'close': close})

    def test_returns_expr(self):
        expr = pl_tmo("open", "close")
        assert isinstance(expr, pl.Expr)

    def test_has_tmo_column(self, sample_df):
        result = sample_df.select(pl_tmo("open", "close"))
        assert "TMO" in result.columns

    def test_struct_has_all_fields(self, sample_df):
        result = sample_df.select(pl_tmo("open", "close"))
        tmo = result["TMO"]
        assert "TMO_14_5_3" in tmo.struct.fields
        assert "TMOs_14_5_3" in tmo.struct.fields
        assert "TMOM_14_5_3" in tmo.struct.fields
        assert "TMOMs_14_5_3" in tmo.struct.fields

    def test_has_valid_values(self, sample_df):
        result = sample_df.select(pl_tmo("open", "close"))
        tmo = result["TMO"]
        main = tmo.struct.field("TMO_14_5_3")
        assert main[30:].null_count() == 0

    def test_offset_parameter(self, sample_df):
        result = sample_df.select(pl_tmo("open", "close", offset=5))
        tmo = result["TMO"]
        main = tmo.struct.field("TMO_14_5_3")
        assert main[:5].null_count() == 5

    def test_lazy_execution(self, sample_df):
        result = sample_df.lazy().select(pl_tmo("open", "close")).collect()
        assert "TMO" in result.columns

    def test_custom_lengths(self, sample_df):
        result = sample_df.select(pl_tmo("open", "close", tmo_length=20, calc_length=10, smooth_length=5))
        tmo = result["TMO"]
        assert "TMO_20_10_5" in tmo.struct.fields

    def test_momentum_enabled(self, sample_df):
        result = sample_df.select(pl_tmo("open", "close", momentum=True))
        tmo = result["TMO"]
        mom_main = tmo.struct.field("TMOM_14_5_3").drop_nulls()
        # When momentum=True, values should not be all zeros
        assert len(mom_main) > 0
        assert not np.isnan(mom_main.to_numpy()).all()  # Check not all NaN
        non_zero = mom_main.to_numpy()[~np.isnan(mom_main.to_numpy())]
        if len(non_zero) > 1:
            assert np.std(non_zero) > 0  # Should have variance

    def test_normalize(self, sample_df):
        result = sample_df.select(pl_tmo("open", "close", normalize=True))
        tmo = result["TMO"]
        main = tmo.struct.field("TMO_14_5_3").drop_nulls()
        # Normalized values should be within [-100, 100] generally
        assert main.max() <= 110  # Allow some margin
        assert main.min() >= -110

    def test_exclusive_vs_inclusive(self, sample_df):
        result_excl = sample_df.select(pl_tmo("open", "close", exclusive=True))
        result_incl = sample_df.select(pl_tmo("open", "close", exclusive=False))
        
        main_excl = result_excl["TMO"].struct.field("TMO_14_5_3")
        main_incl = result_incl["TMO"].struct.field("TMO_14_5_3")
        
        # Results should be different
        valid_excl = main_excl.drop_nulls().to_numpy()
        valid_incl = main_incl.drop_nulls().to_numpy()
        
        if len(valid_excl) > 0 and len(valid_incl) > 0:
            assert not np.allclose(valid_excl[:len(valid_incl)], valid_incl[:len(valid_excl)])

    def test_with_null_values(self):
        df = pl.DataFrame({
            'open': [100.0, None, 102.0] + [100.0 + i * 0.1 for i in range(60)],
            'close': [101.0, 102.0, None] + [101.0 + i * 0.1 for i in range(60)]
        })
        result = df.select(pl_tmo("open", "close"))
        assert result.height == 63

    def test_numerical_parity(self):
        """Verify numerical parity with Pandas implementation."""
        pytest.skip("Pandas implementation removed in Phase 4 purge")
        import pandas as pd
        # from polars_ti.momentum.tmo import tmo as pandas_tmo  # REMOVED: pandas func removed
        
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        open_ = close - np.random.randn(n) * 0.3
        
        pdf = pd.DataFrame({'open': open_, 'close': close})
        pldf = pl.DataFrame({'open': open_, 'close': close})
        
        pandas_result = pandas_tmo(pdf['open'], pdf['close'],
                                   tmo_length=14, calc_length=5, smooth_length=3)
        polars_result = pldf.select(pl_tmo("open", "close",
                                           tmo_length=14, calc_length=5, smooth_length=3))
        
        tmo = polars_result["TMO"]
        pandas_main = pandas_result["TMO_14_5_3"].to_numpy()[30:]
        polars_main = tmo.struct.field("TMO_14_5_3").to_numpy()[30:]
        
        valid_mask = ~np.isnan(pandas_main) & ~np.isnan(polars_main)
        max_diff = np.max(np.abs(pandas_main[valid_mask] - polars_main[valid_mask]))
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance"
