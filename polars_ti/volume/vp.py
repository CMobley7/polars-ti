# -*- coding: utf-8 -*-
# =============================================================================
# Polars VP (Volume Profile) Implementation
# =============================================================================
import polars as pl


def pl_vp(
    df: pl.DataFrame,
    close: str = "close",
    volume: str = "volume",
    width: int = 10,
    sort: bool = False,
) -> pl.DataFrame:
    """Polars: Volume Profile (VP)

    Calculates Volume Profile by slicing price into ranges and aggregating volume.

    Note: This function takes a DataFrame and returns an aggregated DataFrame
    with `width` rows (different from input length).

    Args:
        df: Input Polars DataFrame with close and volume columns
        close: Column name for close prices. Default: "close"
        volume: Column name for volume. Default: "volume"
        width: Number of price ranges/bins. Default: 10
        sort: If True, bin by price ranges. If False, split chronologically. Default: False

    Returns:
        pl.DataFrame: Volume profile with columns:
            - low_close: Lower price bound of range
            - mean_close: Mean price in range
            - high_close: Upper price bound of range
            - pos_volume: Volume on up moves
            - neg_volume: Volume on down moves
            - total_volume: Total volume in range
    """
    if df is None or df.height < width:
        return None

    # Add signed volume columns based on price direction
    df_with_sign = df.with_columns(
        [
            pl.col(close).diff().sign().alias("_sign"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("_sign") > 0).then(pl.col(volume)).otherwise(0.0).alias("_pos_vol"),
            pl.when(pl.col("_sign") < 0).then(pl.col(volume)).otherwise(0.0).alias("_neg_vol"),
        ]
    )

    if sort:
        # Bin by price ranges using Polars cut
        price_min = df.get_column(close).min()
        price_max = df.get_column(close).max()

        # Create bin edges
        bin_edges = [price_min + (price_max - price_min) * i / width for i in range(width + 1)]

        # Use cut to create bins
        df_binned = df_with_sign.with_columns(
            [pl.col(close).cut(bin_edges[1:-1], labels=[str(i) for i in range(width)]).alias("_bin")]
        )

        # Group by bin and aggregate
        result = (
            df_binned.group_by("_bin", maintain_order=True)
            .agg(
                [
                    pl.col(close).min().alias("low_close"),
                    pl.col(close).mean().alias("mean_close"),
                    pl.col(close).max().alias("high_close"),
                    pl.col("_pos_vol").sum().alias("pos_volume"),
                    pl.col("_neg_vol").sum().alias("neg_volume"),
                ]
            )
            .drop("_bin")
            .with_columns([(pl.col("pos_volume") + pl.col("neg_volume")).alias("total_volume")])
        )
    else:
        # Split chronologically - add row index and divide into chunks
        n = df.height
        df_indexed = df_with_sign.with_row_index("_idx")

        # Create chunk assignments
        df_chunked = df_indexed.with_columns([(pl.col("_idx") * width // n).cast(pl.Int32).alias("_chunk")])

        # Group by chunk and aggregate
        result = (
            df_chunked.group_by("_chunk", maintain_order=True)
            .agg(
                [
                    pl.col(close).min().alias("low_close"),
                    pl.col(close).mean().alias("mean_close"),
                    pl.col(close).max().alias("high_close"),
                    pl.col("_pos_vol").sum().alias("pos_volume"),
                    pl.col("_neg_vol").sum().alias("neg_volume"),
                ]
            )
            .drop("_chunk")
            .with_columns([(pl.col("pos_volume") + pl.col("neg_volume")).alias("total_volume")])
        )

    return result
