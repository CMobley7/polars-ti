# Rolling kernel audit and reproduction

The recorded benchmark checkpoint is `cf108754f40b49010723cfce85e1c6c7dbd29abd`,
using Python 3.14.7 and the versions in `REPORT.md`. Current supported versions
are documented in [compatibility](../../docs/compatibility.md).

Run from the `polars-ti` repository on Python 3.14.7:

```bash
uv sync --extra test --dev
uv run python -m scripts.rolling_kernel_audit.audit --baseline > scripts/rolling_kernel_audit/before.json
uv run python -m scripts.rolling_kernel_audit.audit > scripts/rolling_kernel_audit/after.json
uv run python -m scripts.rolling_kernel_audit.benchmark
uv run python -m scripts.rolling_kernel_audit.expressions
uv run python -m scripts.rolling_kernel_audit.accuracy
uv run python -m scripts.rolling_kernel_audit.fir_stress
uv run python -m scripts.rolling_kernel_audit.crossover
./scripts/check.sh
POLARS_TI_SIMULATE_NO_TALIB=1 uv run pytest tests/ -q
```

`baseline.tar.gz` preserves every production source file at the SHA in
`baseline_commit.sha`, before any edits. The reference loader extracts individual
unchanged modules into the ignored `.reference` directory; it does not replace
installed production modules. The AST baseline also reads original tests and
scripts from that recorded Git commit. Baseline expression comparisons are limited
to modules whose relevant calculations are local; they do not claim to reproduce
an entire old dependency graph or research experiment.

`before.json` and `after.json` enumerate every function, its file and line, and
positive or negative findings. This intentionally conservative syntax sweep is
not a complexity proof: disjoint scans, bounded loops, periodic rebuilding, and
arbitrary user callbacks require the classification in `REPORT.md`. It searches
nested loops, slice/aliased-window calls, direct convolution, rolling callbacks,
and the independently identified parameter-sized expression/Pascal construction.

`benchmark.json` contains every raw timing sample, median, relative speed change,
window exponent, and finite-output deviation. `expressions.json` includes Polars
and callback overhead. Each old/new pair uses the same installed runtime, data,
and process; JIT compilation is excluded and measurement order alternates. These
are synthetic kernel timings, not estimates of research-grid wall-clock time.
The machine is shared with other tasks; measurements are not exclusive-host results.

`accuracy.json` measures a separate `math.fsum` two-pass oracle, including
large-offset/tiny-spread data, abrupt scale changes, and a 120,000-row drift test
with windows up to 23,040. The long run samples 256 output windows per configuration.
`crossover.json` separately measures CRSI/MAD at windows 1,024/2,048/8,192
on 32,768 rows. `nan_changes.json` separates intentional bug fixes from floating-point reorderings.
Permanent tests include randomized finite values, constants, NaN warmup prefixes,
interior missing values, ties, empty inputs, oversized windows, and infinities.
No existing test tolerances or golden fixtures were changed.
`test_rolling_fir.py` also verifies cancellation does not trigger a per-window direct scan;
extreme constant values and null-versus-NaN masks have dedicated regressions.

Precision fallbacks preserve direct evaluation for ill-conditioned windows. Their
worst case remains O(n·w); the normal-path complexity and measured corpus behavior
must not be presented as an unconditional asymptotic guarantee. Extended-precision
FIR acceleration requires `longdouble` wider than `float64`; other platforms use
its accurate direct path.
