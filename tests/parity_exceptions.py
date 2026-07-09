# -*- coding: utf-8 -*-
"""Parity-Exceptions registry — the single source of truth for columns that must
NOT simply match the OLD library.

This module is *data only*: a dict keyed by OLD column name -> exception spec.
The parity tests consult it to decide the expected verdict and tolerance for a
column instead of the default "match the OLD golden" behaviour.

Modes:
  * ``match``         — (default; not usually listed here) must match OLD golden.
  * ``match_talib``   — must match the direct TA-Lib reference
                        (``talib_reference.parquet``), NOT the OLD golden,
                        because the OLD native library was wrong here and NEW is
                        correct (COMPARISON_REPORT §4).
  * ``match_canonical`` — must match a documented canonical value (e.g. OBV[0]=0)
                        rather than OLD's buggy behaviour.
  * ``intentional``   — NEW deliberately diverges from OLD by a documented
                        convention (ddof, seed, normalization). Not graded
                        against OLD; pinned by its own dedicated test if at all.
  * ``broken_todo``   — NEW is currently broken/missing/all-NaN (COMPARISON_REPORT
                        §3, §6). A forthcoming parity test should ``xfail`` on
                        these today (they are fixed in WS2–WS4), so the suite
                        surfaces them rather than erroring.

Each entry carries ``tol`` (abs/rel float tolerance or None for exact) and a
``note`` documenting the rationale (with a COMPARISON_REPORT section reference).
"""

from __future__ import annotations

# --- "NEW is better": pin to TA-Lib, not OLD (COMPARISON_REPORT §4) ----------
_MATCH_TALIB = {
    "WCP": "old native WCLPRICE off by ~1337; NEW ~exact vs TA-Lib (§4)",
    "MFI_14": "old native MFI off by ~8.94; NEW ~3e-13 vs TA-Lib (§4)",
    "ADX_14": "old native ADX off by ~6.96; NEW ~0.19, much closer to TA-Lib (§4)",
    # classic port: native +DM/-DM now use Wilder sum-smoothing (== TA-Lib
    # PLUS_DM/MINUS_DM) instead of OLD's average-scale ma('rma'), which diverged
    # by tens of points. Both dm() and adx() emit these; native == talib to ~1e-13
    # (pinned by tests/momentum/test_dm_polars.py + tests/trend/test_adx_polars.py).
    "DMP_14": "classic port: native Wilder sum-smoothing == talib.PLUS_DM (OLD ma('rma') diverged ~44)",
    "DMN_14": "classic port: native Wilder sum-smoothing == talib.MINUS_DM (OLD ma('rma') diverged ~68)",
    # classic port: native CMO now Wilder-smooths gains/losses (CMO == 2*RSI-100,
    # == TA-Lib CMO to ~1e-13) instead of OLD's flat rolling sum, which diverged
    # by ~63 (pinned by tests/momentum/test_cmo_polars.py).
    "CMO_14": "classic port: native Wilder smoothing == talib.CMO (OLD rolling-sum diverged ~63)",
    "RSI_14": "old native RSI off by ~18; NEW ~8.3, closer to TA-Lib (§4)",
    "TRIX_30_9": "old native TRIX off by ~1.8e-3; NEW ~3e-13 vs TA-Lib (§4)",
    # EMA-seed alignment: the native EMA warmup seed was corrected to match
    # TA-Lib on leading-NaN/cascaded inputs (was 1 bar early). DEMA is a pure
    # triple-of-EMA transform, so native DEMA now equals talib.DEMA; the OLD
    # native golden baked in the 1-bar-early seed, so DEMA_10 is graded against
    # the TA-Lib reference (native == talib to ~1e-9, pinned by
    # tests/overlap/test_native_talib_alignment.py).
    "DEMA_10": "native EMA seed now matches TA-Lib (was 1-bar-early); native == talib.DEMA (§4)",
    # Wilder/RMA seed alignment: the native RMA warmup seed was corrected to
    # average the first ``length`` FINITE true-range values at index ``length``,
    # matching TA-Lib's ATR warmup (TR[0] is excluded). Native ATR/NATR now equal
    # talib.ATR/NATR to float noise; the OLD native golden baked in the 1-bar-early
    # seed, so they are graded vs the TA-Lib reference (native == talib to ~1e-9,
    # pinned by tests/overlap/test_native_talib_alignment.py).
    "ATRr_14": "native Wilder/RMA seed now matches TA-Lib ATR warmup; native == talib.ATR (§4)",
    "NATR_14": "native Wilder/RMA seed now matches TA-Lib ATR warmup; native == talib.NATR (§4)",
    # 3 candlestick patterns: OLD native was wrong by 100, NEW matches TA-Lib.
    "CDL_HIGHWAVE": "OLD wrong by 100; NEW == TA-Lib exactly (§4)",
    "CDL_RICKSHAWMAN": "OLD wrong by 100; NEW == TA-Lib exactly (§4)",
    "CDL_SPINNINGTOP": "OLD wrong by 100; NEW == TA-Lib exactly (§4)",
    # classic 9258bf6: cdl_doji averages the prior bars' HL range (.shift(1))
    # and uses <=, so it now matches TA-Lib's CDLDOJI exactly. OLD compared
    # against the current-bar average (look-ahead) and used <.
    "CDL_DOJI_10_0.1": "classic 9258bf6: shifted HL-range avg + <=; NEW == talib.CDLDOJI exactly (OLD looked ahead)",
}

# --- Intentional convention divergences from OLD (COMPARISON_REPORT §7) -------
_INTENTIONAL = {
    "close_Z_30_1": "NEW ddof=0 (population std, TA-Lib style) vs OLD ddof=1 (§7)",
    "open_Z_30_1": "NEW ddof=0 vs OLD ddof=1 (§7)",
    "high_Z_30_1": "NEW ddof=0 vs OLD ddof=1 (§7)",
    "low_Z_30_1": "NEW ddof=0 vs OLD ddof=1 (§7)",
    "ZS_30": "NEW ddof=0 vs OLD ddof=1 (§7)",
    # stdev/variance now default ddof=0 (population, TA-Lib/TradingView/Bollinger
    # convention) instead of ddof=1 (which pandas-ta only used to mirror pandas'
    # .std() default). Native now equals talib.STDDEV/VAR; the OLD native golden
    # baked in ddof=1. Consistent with the z-score ddof=0 choice above.
    "STDEV_30": "NEW ddof=0 (population, TA-Lib style) default vs OLD ddof=1 (§7)",
    "VAR_30": "NEW ddof=0 (population, TA-Lib style) default vs OLD ddof=1 (§7)",
    "PVI": "NEW seeds initial=100 (StockCharts canonical) vs OLD first-close (§7)",
    "PVIe_255": "NEW seeds initial=100 vs OLD first-close (§7)",
    "MASSI_9_25": "NEW NaN-skipping cascaded EMA vs OLD nested TA-Lib EMA (§7)",
    "OBVe_4": "NEW OBV[0]=0 canonical seed vs OLD signed_series ignoring initial (§7)",
    "OBVe_12": "NEW OBV[0]=0 canonical seed vs OLD signed_series ignoring initial (§7)",
    # WS3: OLD aberration never propagated talib to its internal atr(), so the
    # `old_notalib` golden for these bands actually captured TA-Lib ATR values
    # (an OLD propagation bug). NEW propagates talib, so native ABER ATR bands
    # are correct pandas-ta and intentionally diverge from the buggy native
    # golden; they DO match the TA-Lib reference in talib mode (§5/Decision 4).
    "ABER_ATR_5_15": "OLD native golden is TA-Lib ATR (OLD talib non-propagation bug); NEW native = pandas-ta (§5)",
    "ABER_SG_5_15": "OLD native golden is TA-Lib ATR (OLD talib non-propagation bug); NEW native = pandas-ta (§5)",
    "ABER_XG_5_15": "OLD native golden is TA-Lib ATR (OLD talib non-propagation bug); NEW native = pandas-ta (§5)",
    # OLD vhm divided by ``pstdev(volume, slength)`` — but statistics.pstdev's
    # 2nd arg is the assumed MEAN, not a window, so OLD's denominator is a single
    # constant (~3.9e6) for the whole series. NEW uses a proper rolling stdev
    # (canonical TradingView "Heatmap Volume"); it intentionally diverges from
    # the buggy OLD golden and is pinned by test_vhm_canonical (not vs OLD).
    "VHM_610": "OLD pstdev(volume, slength) misuses mean-arg as window (constant denom); NEW rolling stdev is canonical",
    # classic 1474768: VIDYA now seeds the recurrence with the SMA of the first
    # `length` closes (vidya[length-1] = mean(close[:length])) instead of 0; the
    # zero-seed produced a long, materially-wrong transient. There is no TA-Lib
    # VIDYA, so it is graded against the classic fork's vidya output (native CMO)
    # by tests/overlap/test_vidya_polars.py (max_abs ~5.7e-14), not the OLD
    # goldens (which baked in the zero-seed bug). NEW diverges in BOTH modes.
    "VIDYA_14": "classic 1474768: SMA-seed recurrence; no TA-Lib equiv; pinned vs classic fork (native), not OLD golden",
    # classic 9258bf6: PSAR reversal test now uses the GUARDED SAR (guard spans
    # the prior TWO bars' high/low) and long/short are reclassified from the
    # combined SAR using close (SAR<close->long, else short), matching TA-Lib.
    # TA-Lib's SAR is a single series with no af/reversal columns, so these can't
    # be folded onto talib_reference; the combined SAR line is validated against
    # talib.SAR (== exactly after the warmup transient) by
    # tests/trend/test_psar_polars.py. NEW diverges from the OLD goldens (which
    # baked in the pre-fix reversal/guard logic) in BOTH modes.
    "PSARl_0.02_0.2": "classic 9258bf6: guarded reversal + 2-bar guard + close reclassify; combined SAR == talib.SAR (pinned by test)",
    "PSARs_0.02_0.2": "classic 9258bf6: guarded reversal + 2-bar guard + close reclassify; combined SAR == talib.SAR (pinned by test)",
    "PSARaf_0.02_0.2": "classic 9258bf6: acceleration factor follows the TA-Lib-aligned reversal logic; no TA-Lib equivalent column",
    "PSARr_0.02_0.2": "classic 9258bf6: reversal flag follows the TA-Lib-aligned guarded reversal; no TA-Lib equivalent column",
    # EMA-seed alignment: the native EMA warmup seed was corrected to match
    # TA-Lib on leading-NaN/cascaded inputs (was 1 bar early). These indicators
    # feed a leading-NaN diff/signed series into the native EMA, so their native
    # output shifted to match the SAME indicator's talib=True output. There is no
    # TA-Lib reference column for them, so they are excluded from OLD-golden
    # grading and pinned native==talib by tests/overlap/test_native_talib_alignment.py.
    "ZL_EMA_10": "native EMA seed now matches TA-Lib (was 1-bar-early); no TA-Lib equiv; pinned native==talib",
    "TSI_13_25_13": "native EMA seed now matches TA-Lib (was 1-bar-early); no TA-Lib equiv; pinned native==talib",
    "TSIs_13_25_13": "native EMA seed now matches TA-Lib (was 1-bar-early); no TA-Lib equiv; pinned native==talib",
    "TMO_14_5_3": "native EMA seed now matches TA-Lib (was 1-bar-early); no TA-Lib equiv; pinned native==talib",
    "TMOs_14_5_3": "native EMA seed now matches TA-Lib (was 1-bar-early); no TA-Lib equiv; pinned native==talib",
    "EFI_13": "native EMA seed now matches TA-Lib (was 1-bar-early); no TA-Lib equiv; pinned native==talib",
    "KVO_34_55_13": "native EMA seed now matches TA-Lib (was 1-bar-early); no TA-Lib equiv; pinned native==talib",
    "KVOs_34_55_13": "native EMA seed now matches TA-Lib (was 1-bar-early); no TA-Lib equiv; pinned native==talib",
}

# --- WS2-WS4 FIXED: now match the OLD golden within float tol -----------------
# These were broken/missing/all-NaN at the start of remediation and have since
# been repaired. They are promoted out of ``broken_todo`` to the default "match"
# verdict and ENFORCED (strict, non-xfail) by ``test_fixed_indicators_parity`` so
# any regression turns the suite red. Kept here as data for that test and as a
# record of what WS2-WS4 delivered.
_FIXED_GROUPS = {
    # WS2 — Expr-API misuse / restored output (§3)
    "ha": ["HA_open", "HA_high", "HA_low", "HA_close"],
    "ichimoku": ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"],
    "mama_fama": ["MAMA_0.5_0.05", "FAMA_0.5_0.05"],
    "hilo": ["HILO_13_21", "HILOl_13_21", "HILOs_13_21"],
    "pivots": [
        "PIVOTS_TRAD_D_P",
        "PIVOTS_TRAD_D_S1",
        "PIVOTS_TRAD_D_S2",
        "PIVOTS_TRAD_D_S3",
        "PIVOTS_TRAD_D_S4",
        "PIVOTS_TRAD_D_R1",
        "PIVOTS_TRAD_D_R2",
        "PIVOTS_TRAD_D_R3",
        "PIVOTS_TRAD_D_R4",
    ],
    "pmax": ["PMAX_10_3.0", "PMAXd_10_3.0", "PMAXl_10_3.0", "PMAXs_10_3.0"],
    "rwi": ["RWIh_14", "RWIl_14"],
    # WS2 — talib kwarg crash repaired (§3)
    "macd": ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
    "ppo": ["PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9"],
    # WS2 — present-but-all-NaN repaired (§3)
    "atrts": ["ATRTSe_14_20_3.0"],
    "chop": ["CHOP_14_1_100.0"],
    "cksp": ["CKSPl_10_3_20", "CKSPs_10_3_20"],
    "halftrend_tl": ["HT_TL"],  # trend line fixed; the other HT_* still broken
    # WS4 — NaN-comparison guard in increasing()/decreasing() (NaN > 0 == True
    # in Polars produced spurious warmup 1s); now 0 during warmup like pandas.
    "amat": ["AMATe_LR_8_21_2", "AMATe_SR_8_21_2"],
    "aobv": ["AOBV_LR_2", "AOBV_SR_2"],
    # WS4 — restored the [-1, 1] input remap (raw prices saturated exp() to ≈1)
    # and the missing INVFISHERs signal column.
    "invfisher": ["INVFISHER_1.0", "INVFISHERs_1.0"],
    # WS2 — accessor passed close into the `length` arg; fixed dispatch. Also
    # made rvi honour talib (TA-Lib STDDEV/EMA) so both modes match OLD.
    "rvi": ["RVI_14"],
    "inertia": ["INERTIA_20_14"],  # depends on rvi; fixed once rvi was fixed
    "vfi": ["VFI_130"],  # accessor passed high/low into close/volume/length args
    "pgo": ["PGO_14"],  # hardcoded talib=False; now honours talib (TA-Lib SMA/ATR/EMA)
    "avsl": ["AVSL_12_26"],  # accessor passed high/low/close into close/low/volume args
    # WS2 — accessor passed close into the `legs` arg; fixed dispatch.
    "zigzag": ["ZIGZAGd_5.0%_10", "ZIGZAGs_5.0%_10", "ZIGZAGv_5.0%_10"],
    # WS2 — accessor passed volume into the `length` arg; int-default naming
    # (1/50 not 1.0/50.0); reuse talib-aware atr()/rsi() (mamode='sma') so
    # talib mode matches OLD exactly. (Native residual tracks OLD's buggy native
    # RSI seed — see RSI_14 match_talib exception.)
    "alphatrend": ["ALPHAT_14_1_50", "ALPHATl_14_1_50_2"],
    # WS2 — accessor passed high/low into the close/length args; trixh takes
    # only close. Now emits TRIX/TRIXs/TRIXh_18_9.
    "trixh": ["TRIXh_18_9"],
    # WS2 — were still the OLD callable/DataFrame API (dropped by _post_process);
    # rewritten as native Exprs with their numba recursion kernels.
    "ebsw": ["EBSW_40_10"],
    "reflex": ["REFLEX_20_20_0.04"],
    # WS2 — struct fields renamed to the OLD flat convention so the parity engine
    # folds them; internal atr() now honours talib (OLD always used TA-Lib ATR).
    # (HT_direction is a string column, enforced by test_halftrend_direction_parity.)
    "halftrend": [
        "HT_atr_high_14_2_2",
        "HT_atr_low_14_2_2",
        "HT_close_14_2_2",
        "HT_arr_up_14_2_2",
        "HT_arr_down_14_2_2",
    ],
    # WS5 — surfaced once the struct/rename name-map (tests/_parity.RENAME_MAP)
    # let the parity engine grade these columns. Each is graded through the map.
    # qqe: result was never aliased (collided as 'close'), _props had a leading
    #   underscore and replaced '.'; restored naming + the RSIMA basis column.
    "qqe": [
        "QQE_14_5_4.236",
        "QQE_14_5_4.236_RSIMA",
        "QQEl_14_5_4.236",
        "QQEs_14_5_4.236",
    ],
    # cube: emitted only the main line; now returns the CUBEs signal too.
    "cube": ["CUBE_3.0_-1", "CUBEs_3.0_-1"],
    # kc/thermo/chandelier: internal ma()/atr() hardcoded talib=False; now honour
    # talib (OLD used TA-Lib). thermo/chandelier also had NaN-comparison flag bugs.
    "kc": ["KCLe_20_2", "KCBe_20_2", "KCUe_20_2"],
    "thermo": [
        "THERMO_20_2_0.5",
        "THERMOma_20_2_0.5",
        "THERMOl_20_2_0.5",
        "THERMOs_20_2_0.5",
    ],
    "chandelier_exit": [
        "CHDLREXTl_22_22_14_2.0",
        "CHDLREXTs_22_22_14_2.0",
        "CHDLREXTd_22_22_14_2.0",
    ],
    # vwap: accessor now passes a datetime column so VWAP anchors (daily reset).
    "vwap": ["VWAP_D"],
    # Folded-and-already-correct struct renames (guard the name-map folding).
    "aroon": ["AROOND_14", "AROONU_14", "AROONOSC_14"],
    "fvg": ["FVGh_0", "FVGl_0", "FVGt_0"],
    "hwc": ["HWL_1", "HWM_1", "HWU_1"],
    "tos_stdevall": [
        "TOS_STDEVALL_LR",
        "TOS_STDEVALL_L_1",
        "TOS_STDEVALL_L_2",
        "TOS_STDEVALL_L_3",
        "TOS_STDEVALL_U_1",
        "TOS_STDEVALL_U_2",
        "TOS_STDEVALL_U_3",
    ],
    # accessor passed (high, low, close) but smc()'s first param is open_, so
    # all OHLC inputs were shifted; fixed the dispatch (matches both modes).
    "smc": [
        "SMChv_14_50_20_5",
        "SMCbf_14_50_20_5",
        "SMCbi_14_50_20_5",
        "SMCbp_14_50_20_5",
        "SMCtf_14_50_20_5",
        "SMCti_14_50_20_5",
        "SMCtp_14_50_20_5",
    ],
}

# --- Still broken / missing / all-NaN in NEW (COMPARISON_REPORT §3, §6) --------
# A parity test xfails on these today; they are remediated in WS2 (restore broken
# indicators), WS3 (presma seeding), WS4 (discrete bugs). As each is fixed it is
# moved up into ``_FIXED_GROUPS`` and enforced.
_BROKEN_TODO_GROUPS = {
    # (empty) — every catalogued WS2–WS4 defect is fixed and enforced above.
    # WS3 FIXED — the presma seed bug (§5) is repaired; these now match the OLD
    # native golden within float tol and are enforced by the parity suite:
    #   ATRr_14, NATR_14, DEMA_10, ZL_EMA_10, ABER_ZG_5_15 -> default "match".
    #   ABER_ATR/SG/XG_5_15 -> "intentional" (registered above; OLD native golden
    #   captured TA-Lib ATR via an OLD talib non-propagation bug).
}


def _build() -> dict[str, dict]:
    reg: dict[str, dict] = {}
    for col, note in _MATCH_TALIB.items():
        # candle/flag columns are exact; floats use default tol.
        exact = col.startswith("CDL_")
        reg[col] = {"mode": "match_talib", "tol": None if exact else 1e-6, "note": note}
    for col, note in _INTENTIONAL.items():
        reg[col] = {"mode": "intentional", "tol": None, "note": note}
    for group, cols in _BROKEN_TODO_GROUPS.items():
        for col in cols:
            # Keep the first registration if a column appears twice; broken_todo
            # never overrides a match_talib/intentional verdict.
            if col in reg:
                continue
            reg[col] = {"mode": "broken_todo", "tol": None, "note": f"WS2-4 fix: {group} (§3/§5/§6)"}
    return reg


#: OLD-column-name -> {mode, tol, note}
PARITY_EXCEPTIONS: dict[str, dict] = _build()

#: WS2-WS4-repaired columns, enforced (strict) against the OLD golden. These are
#: NOT in PARITY_EXCEPTIONS — they take the default "match" verdict — but are
#: listed so the parity suite can pin them explicitly.
FIXED_COLS: list[str] = sorted(col for cols in _FIXED_GROUPS.values() for col in cols)


def mode_for(col: str) -> str:
    """Return the exception mode for ``col``, or ``"match"`` (default)."""
    return PARITY_EXCEPTIONS.get(col, {}).get("mode", "match")
