# Independent review loop

The user initially requested Sol and Terra reviews of the entire changeset, followed by
fixes and repeated review until both report no actionable findings. All reviews
are read-only and target the real `rolling-kernels` working tree.

## Round 1

- Sol (`gpt-5.6-sol`): FIR's output-relative fallback caused O(n·w) work on
  zero-centered signals. The added cancellation scan-count regression failed
  before the fix. The trigger now compares an absolute FFT error estimate with
  a local direct-summation rounding budget; cancellation alone does not cause
  direct scans. NaN windows are masked without evaluating their full windows.
  Focused FIR/fix/expression suite: 188 passed.
- Terra (`gpt-5.6-terra`): exact large constant windows could yield spurious
  variance because sum/divide rounding moves their mean, and squaring the
  residual overflows. Three extreme-constant regressions failed before the
  explicit constant-window handling. Precision/fix suite: 47 passed.
- Terra: `.python-version` was ignored and therefore absent from the proposed
  tracked files. Its ignore rule was removed; the 3.14.7 pin is now included.
- Parent's fresh synthesis: variance/stdev converted all float NaNs to Polars
  nulls. Four mask regressions failed before preserving null windows separately
  from NaN values; all six mask cases now pass.

Review statements are reproduced as findings, not accepted on authority alone.
In particular, the constant-series mechanism is the rounded mean's residual
at extreme scale, not necessarily a shift as large as the original values.

## Round 2

- Fresh Sol (`gpt-5.6-sol`): the FFT error estimate used the maximum of the whole
  series, so a distant outlier could force direct scans long after it expired.
  The new regression failed before independent convolution blocks and block-local
  error estimates were introduced. FIR/expression suite: 180 passed afterward.
- Fresh Terra verification was clean on the round-2 changes.

## Follow-up

The final Astra/Terra verification appears below. The benchmark and full quality gates are refreshed after
any implementation fixes.

## Round 3 and reviewer change

- Fresh Sol found nonconstant high/low-scale moment failures: summing before
  dividing can overflow the mean, while squared deviations can underflow even
  when standard deviation is representable. A new scale-invariance regression
  failed before a power-of-two-scaled, centered direct calculation was added for
  extreme windows. Variance preserves its representable range; standard deviation,
  zscore, skew, kurtosis and regression use normalized intermediates when needed.
  Focused fix/precision/equivalence suite: 427 passed, 54 skipped.
- Terra round 2 was clean before that additional numerical fix.
- The user then replaced Sol with **Astra (`gpt-6-astra`)** for remaining rounds.
  Fresh Astra and Terra reviews must both return clean on the final changeset.

## Astra/Terra round 1

The desktop subagent service reached its thread limit. Reviews continued through
local Codex 0.155.1, explicitly selecting `gpt-6-astra` and `gpt-5.6-terra`;
the runner headers confirmed both model names. The auxiliary resumed audit could
not attest its model and is not counted as an Astra signoff.

- Auxiliary audit: SMA overflow before division and signed-infinity WMA behavior.
  Eight regressions failed before the normalized fallback and infinity handling.
- Astra: descending WMA cancellation. Reversed-input weighted evaluation avoids
  subtracting rounded totals. Nearly flat Trendflex/Reflex amplified cancellation;
  their common primitive now accumulates centered differences. Sample variance
  now applies ddof before rescaling subnormal values. Each had a failing test.
- Astra: finite outliers contaminated WMA/CG and MSW after leaving the window;
  recurrence state now rebuilds after a large downward scale transition. MAVP's
  wrapped query discarded compensation; both ranges now share the accumulator.
  Six new regression cases reproduced these failures before the fixes.
- Terra: exact quantiles evaluated an unused infinite neighbor times zero. Exact
  fractions now return the selected order statistic directly; the regression
  failed before that change. Its exploratory entropy concern was not a changed
  behavior: the preserved implementation uses the same two rolling sums.

Numba caches were cleared after shared-helper edits because existing compiled
callers can retain old helper code. Final gates and measurements use freshly
compiled callers. Follow-up reviewers independently evaluate current source.

## Astra/Terra final verification

Follow-ups reproduced and fixed further conditioning cases: moderate outliers
followed by flat/tiny signals in MSW, expired outliers followed by cancelling
linear sums, and overflow inside MAVP tree nodes. Public full-window SMA now
uses the corrected kernel. Its null mask is preserved; smaller min_periods
continues using Polars. Binary scaling replaced arbitrary-maximum normalization
after an exact-cancellation counterexample. WMA/CG now normalize before computing
representable results whose unscaled intermediate weights overflow. Every fix has
a permanent red/green regression; all 56 precision regressions pass.

- **Astra (`gpt-6-astra`), final review:** “Clean: no significant actionable
  findings remain in the reviewed changeset.” Independently verified cancellation,
  mixed-scale recovery and the latest WMA/CG fallback. Its rolling suite reported
  **683 passed / 54 skipped**; additional numerical probes stayed within budgets.
- **Terra (`gpt-5.6-terra`), final review:** “Clean. No significant remaining
  actionable findings.” Verified the WMA/CG fallback through public expressions
  and adversarial finite-scale oracles, and **56 precision tests passed**.

Both model identities were confirmed in runner headers. Their source probes
were independent of the parent's cached Numba callers. The parent verified the
quoted findings and fixes against code and permanent tests before accepting the
signoffs. Full fresh-cache quality gates and benchmark artifacts are recorded
in REPORT.md; no existing test tolerances or golden files were weakened.

## Final artifacts and CI verification

- Astra: “Clean: no actionable findings in the specified artifacts or CI changes.”
  Verified all 49 benchmark rows, accuracy summaries, audit counts, unchanged
  crossover references, timing formulas and the full-extras suite result.
- Terra: “Clean. No material artifact, claim, crossover-reference, or CI
  configuration defect found.” Independently checked the JSON/report mappings,
  paired frozen-reference measurements and locked CI dependency resolution.

The full optional environment passed 2,974 tests with 56 skips. All optional
requirements also passed pip-audit with no known vulnerabilities. This final
artifact-only review followed both reviewers' clean numerical signoffs.
