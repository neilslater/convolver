# Selection calibration

This repository-only diagnostic harness measures the current public direct, FFT
and automatic methods. It is not loaded at runtime or shipped in the gem.
After compiling a release build, collect separate processes:

```sh
bundle exec ruby -Ilib benchmark/recalibration.rb > training.json
RECALIBRATION_SUITE=holdout bundle exec ruby -Ilib benchmark/recalibration.rb > holdout.json
RECALIBRATION_SUITE=validation bundle exec ruby -Ilib benchmark/recalibration.rb > validation.json
```

Repeat each command in a fresh process without concurrent benchmarks. The
153-row training and 105-row holdout suites include both floating result types,
both operations, integer conversion, views, ranks zero through five, boundary
modes, awkward shapes, periodic folding, clear wins and crossover cases. Shapes
were declared before the v4 calibration fit. For future recalibration, reserve
new shapes too: these holdouts have already been used for v4 model validation.
The 32-row `validation` suite adds nearby 2D crossovers declared after the
Intel-driven x86 setup correction was frozen. Report it separately from the
older regression suites; it was not used to choose the correction.

Each case checks dtype, shape and numerical agreement before measuring complete
public calls. Seven interleaved batches target 6 ms each, clamped to 3..400 calls,
with normal GC. JSON contains all batch samples, medians, ranges, selected paths,
estimates, separate allocation counts, work counts and CPU/build metadata.
`cost_profile` records the current platform's table; scalar and higher-rank cases
still use the legacy fallback. Compare automatic time to the faster explicit
path per row and review subgroups and repeated outliers, not just pooled means.
Timings are review evidence, never correctness pass/fail assertions.

Component probes use prepared inputs and reusable plans/operations to isolate
work. They omit setup that complete calls perform, so their timings cannot be
summed into a whole-call estimate. Folding preparation includes circular
operation construction and required casts. Linear FFT execution includes padding
and result materialization. Keep lint/sanitizer/coverage builds out of timings.

The **Selection calibration** GitHub workflow can be run manually. It collects
two fresh processes per suite on Ubuntu 24.04 x86/arm64 and Ubuntu 22.04 x86,
using Ruby 4.0.6. During the v4 implementation it also runs for relevant changes
on the `recalibrate-auto-selection` PR branch. Other PRs do not automatically
consume this additional runner matrix. Normal CI retains the shorter selection
benchmark. Reports expire after 14 days; preserve essential findings and limits
in a decision record before they expire.

The production coefficient tables live in `lib/convolver/cost_profiles.rb`.
Do not maintain a second copy here. The v4 study's frozen prototypes and 0.8/0.9
margin comparison are preserved at commit `16f27dd`; use that historical checkout
to reproduce the original study. The maintained harness now tests actual public
entry points. Implementation validation subsequently raised the x86 rank-two
FFT setup cost from 87 to 120 microseconds after an Intel runner exposed
repeatable thin-array and small-2D outliers. Other profile terms stayed fixed.
Profile detection, selection regressions, lower-bound inequalities,
overflow, dtype and error behavior are covered by the normal `spec/` suite.
