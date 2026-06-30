# -*- coding: utf-8 -*-
"""Parity engine: compare a NEW (Polars) study output against a golden parquet.

This is a *reusable module*, NOT a test file (it is not collected by pytest; it
exposes helpers the parity tests import).

It ports the verified logic from ``tmp/compare/compare.py``:
  * flatten NEW Polars struct columns into dotted ``parent.field`` names,
  * fold NEW (flattened/dotted) names onto OLD/golden names via suffix-matching
    (struct keys AND fields can both contain '.', so a naive ``split('.')`` is
    unreliable; we match by exact name first, then longest suffix),
  * align rows by ``date``,
  * per column compute ``overlap / null_mismatch / max_abs / max_rel`` treating
    NaN as missing (Polars NaN is not null), and classify the column.

Classifications (``cls``):
  * ``match``        — values within tolerance AND null masks agree.
  * ``warmup/null``  — values within tolerance but null masks differ (warmup).
  * ``material``     — overlapping values exceed tolerance.
  * ``no-overlap``   — no row where both sides are non-NaN.
  * ``non-numeric``  — at least one side is not numeric-coercible.

Public API:
  * ``flatten_structs(df)``
  * ``compare_frames(new_df, golden_df, abs_tol=1e-6, rel_tol=1e-6)`` ->
        ``{old_col: {new_col, overlap, null_mismatch, max_abs, max_rel, cls}}``
  * ``assert_column(report, col, abs_tol=1e-6, rel_tol=1e-6, exact=False)``
"""

from __future__ import annotations

import numpy as np
import polars as pl

# OHLCV/base columns that are inputs, not indicator outputs.
BASE = {"date", "open", "high", "low", "close", "volume", "dividends", "stock splits"}

DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 1e-6


def flatten_structs(df: pl.DataFrame) -> pl.DataFrame:
    """Unnest every Struct column into dotted ``parent.field`` columns."""
    changed = True
    while changed:
        changed = False
        for name, dtype in list(df.schema.items()):
            if isinstance(dtype, pl.Struct):
                fields = dtype.fields
                df = df.with_columns(
                    [pl.col(name).struct.field(f.name).alias(f"{name}.{f.name}") for f in fields]
                ).drop(name)
                changed = True
    return df


def _normalise_date(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a string ``date`` column exists and the frame is sorted by it."""
    if "date" not in df.columns:
        first = df.columns[0]
        df = df.rename({first: "date"})
    return df.with_columns(pl.col("date").cast(pl.Utf8)).sort("date")


def _to_float(series: pl.Series):
    """Return a float64 numpy array (NaN for nulls), or None if not numeric."""
    try:
        if series.dtype == pl.Boolean:
            series = series.cast(pl.Int8)
        arr = series.cast(pl.Float64, strict=False).to_numpy()
        return arr.astype(np.float64)
    except Exception:
        return None


# Authoritative OLD-flat -> NEW-flattened(dotted) name map for the deliberate
# struct/rename divergences (Decision 1). The NEW library keeps struct/dotted
# output as canonical; these columns can't be folded by suffix matching because
# the NEW struct field uses a short/different key, so they are mapped explicitly.
# The values here are the column names produced by ``flatten_structs`` (i.e.
# ``StructName.field``). Keep in sync with the optional pandas-ta-compat layer.
RENAME_MAP: dict[str, str] = {
    # Aroon struct (short field keys)
    "AROOND_14": "AROON_14.AROOND",
    "AROONOSC_14": "AROON_14.AROONOSC",
    "AROONU_14": "AROON_14.AROONU",
    # Chandelier Exit struct
    "CHDLREXTd_22_22_14_2.0": "CHDLREXT_22_22_14_2.0.direction",
    "CHDLREXTl_22_22_14_2.0": "CHDLREXT_22_22_14_2.0.long",
    "CHDLREXTs_22_22_14_2.0": "CHDLREXT_22_22_14_2.0.short",
    # Fair Value Gap struct
    "FVGh_0": "FVG_0.fvg_high",
    "FVGl_0": "FVG_0.fvg_low",
    "FVGt_0": "FVG_0.fvg_type",
    # Holt-Winters channel struct
    "HWL_1": "HWC_1.hwl",
    "HWM_1": "HWC_1.hwm",
    "HWU_1": "HWC_1.hwu",
    # Keltner Channel (ema) struct
    "KCBe_20_2": "KC_e_20_2.kcb",
    "KCLe_20_2": "KC_e_20_2.kcl",
    "KCUe_20_2": "KC_e_20_2.kcu",
    # Thermo struct
    "THERMO_20_2_0.5": "THERMO_20_2_0.5.thermo",
    "THERMOl_20_2_0.5": "THERMO_20_2_0.5.thermo_long",
    "THERMOma_20_2_0.5": "THERMO_20_2_0.5.thermo_ma",
    "THERMOs_20_2_0.5": "THERMO_20_2_0.5.thermo_short",
    # TOS Standard-deviation-all struct (dot vs underscore separator)
    "TOS_STDEVALL_LR": "TOS_STDEVALL.LR",
    "TOS_STDEVALL_L_1": "TOS_STDEVALL.L_1",
    "TOS_STDEVALL_L_2": "TOS_STDEVALL.L_2",
    "TOS_STDEVALL_L_3": "TOS_STDEVALL.L_3",
    "TOS_STDEVALL_U_1": "TOS_STDEVALL.U_1",
    "TOS_STDEVALL_U_2": "TOS_STDEVALL.U_2",
    "TOS_STDEVALL_U_3": "TOS_STDEVALL.U_3",
    # VWAP anchor naming
    "VWAP_D": "VWAP_1D",
}


def _build_name_map(old_inds: list[str], new_inds: list[str]) -> list[tuple[str, str]]:
    """Fold flattened NEW ``structkey.field`` names onto OLD names.

    Explicit :data:`RENAME_MAP` first; then exact match; otherwise the shortest
    NEW column that ends with the OLD name (``== old``, ``endswith('.'+old)`` or
    ``endswith(old)``).
    """
    new_set = set(new_inds)
    common: list[tuple[str, str]] = []
    matched_new: set[str] = set()

    # 1) Explicit rename map (deliberate struct/rename divergences).
    for oc in old_inds:
        nc = RENAME_MAP.get(oc)
        if nc is not None and nc in new_set and nc not in matched_new:
            common.append((oc, nc))
            matched_new.add(nc)

    # 2) Exact name match.
    already = {a for a, _ in common}
    for oc in old_inds:
        if oc in already:
            continue
        if oc in new_set and oc not in matched_new:
            common.append((oc, oc))
            matched_new.add(oc)

    # 3) Suffix fold for the remaining struct columns.
    already = {a for a, _ in common}
    for oc in old_inds:
        if oc in already:
            continue
        cand = None
        for nc in new_inds:
            if nc in matched_new:
                continue
            if nc == oc or nc.endswith("." + oc) or nc.endswith(oc):
                if cand is None or len(nc) < len(cand):
                    cand = nc
        if cand is not None:
            common.append((oc, cand))
            matched_new.add(cand)
    return common


def compare_frames(
    new_df: pl.DataFrame,
    golden_df: pl.DataFrame,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, dict]:
    """Compare a NEW study-output frame against a golden frame.

    Returns a dict keyed by OLD/golden column name -> per-column metrics dict
    ``{new_col, overlap, null_mismatch, max_abs, max_rel, cls}``. Also includes
    the synthetic keys ``__old_only__`` and ``__new_only__`` listing unmatched
    columns on each side.
    """
    new = _normalise_date(flatten_structs(new_df))
    old = _normalise_date(golden_df)

    # Align rows by position after the common date sort (both slices share dates).
    n = min(old.height, new.height)
    old = old.head(n)
    new = new.head(n)

    old_inds = [c for c in old.columns if c not in BASE]
    new_inds = [c for c in new.columns if c not in BASE]
    common = _build_name_map(old_inds, new_inds)
    matched_new = {nc for _, nc in common}
    matched_old = {oc for oc, _ in common}

    report: dict[str, dict] = {}
    for oc, nc in common:
        so, sn = old[oc], new[nc]
        ao, an = _to_float(so), _to_float(sn)
        if ao is None or an is None:
            eq = int((so.cast(pl.Utf8) == sn.cast(pl.Utf8)).sum())
            report[oc] = dict(
                new_col=nc,
                overlap=0,
                null_mismatch=None,
                max_abs=None,
                max_rel=None,
                cls="non-numeric",
                note=f"str_eq={eq}/{n}",
            )
            continue
        null_o = np.isnan(ao)
        null_n = np.isnan(an)
        null_mismatch = int(np.sum(null_o != null_n))
        both = ~null_o & ~null_n
        overlap = int(np.sum(both))
        if overlap == 0:
            report[oc] = dict(
                new_col=nc,
                overlap=0,
                null_mismatch=null_mismatch,
                max_abs=None,
                max_rel=None,
                cls="no-overlap",
                note="",
            )
            continue
        d = np.abs(ao[both] - an[both])
        denom = np.abs(ao[both]) + np.abs(an[both]) + 1e-12
        # inf-valued samples (some indicators emit inf) make inf/inf -> nan here;
        # that is handled by the tolerance check below, so silence the warning.
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = d / denom
        max_abs = float(np.max(d))
        max_rel = float(np.max(rel))
        matched_vals = (max_abs <= abs_tol) or (max_rel <= rel_tol)
        if matched_vals and null_mismatch == 0:
            cls = "match"
        elif matched_vals and null_mismatch > 0:
            cls = "warmup/null"
        else:
            cls = "material"
        report[oc] = dict(
            new_col=nc,
            overlap=overlap,
            null_mismatch=null_mismatch,
            max_abs=max_abs,
            max_rel=max_rel,
            cls=cls,
            note="",
        )

    report["__old_only__"] = sorted(oc for oc in old_inds if oc not in matched_old)
    report["__new_only__"] = sorted(nc for nc in new_inds if nc not in matched_new)
    return report


def assert_column(
    report: dict[str, dict],
    col: str,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    exact: bool = False,
) -> None:
    """Assert a single column in ``report`` is at parity.

    Args:
        report: output of :func:`compare_frames`.
        col: the OLD/golden column name to check.
        abs_tol / rel_tol: float tolerances. Ignored when ``exact=True``.
        exact: when True (signal/flag/candle columns) require max_abs == 0 and
            no null-mask mismatch.
    """
    assert col in report, f"{col!r} not present in parity report (unmatched?)"
    r = report[col]
    cls = r["cls"]
    assert cls != "non-numeric", f"{col}: non-numeric ({r.get('note')})"
    assert cls != "no-overlap", f"{col}: no overlapping non-NaN values (all-NaN?)"
    assert r["overlap"] > 0, f"{col}: zero overlap"
    if exact:
        assert r["max_abs"] == 0, f"{col}: not exact, max_abs={r['max_abs']}"
        assert r["null_mismatch"] == 0, f"{col}: null-mask mismatch={r['null_mismatch']}"
        return
    ok = (r["max_abs"] <= abs_tol) or (r["max_rel"] <= rel_tol)
    assert ok, (
        f"{col}: material diff max_abs={r['max_abs']:.3e} max_rel={r['max_rel']:.3e} "
        f"(tol abs={abs_tol:.1e} rel={rel_tol:.1e}, new_col={r['new_col']})"
    )
