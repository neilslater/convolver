# Recalibration research

This opt-in study leaves `lib/` and `ext/` behavior unchanged. After a release
build, collect separate processes for fitting and validation:

```sh
bundle exec ruby -Ilib benchmark/recalibration.rb > training.json
RECALIBRATION_SUITE=holdout bundle exec ruby -Ilib benchmark/recalibration.rb > holdout.json
```

Repeat each command in a fresh process without concurrent benchmarks. The
dedicated-branch research workflow collects two processes per suite on Linux
x86 and arm64, using Ruby 4.0.6 on both. It runs only for relevant PR changes on
`recalibrate-auto-selection`; regular CI remains unchanged. Reports expire after
14 days, so preserve essential conditions, findings and limitations in the proposal.

`cases.rb` declares training and holdout shapes before fitting. Keep holdout
measurements out of coefficient fitting. `StudyCase` checks numerical agreement,
observes dispatch through an untimed TracePoint, and records allocation counts
outside timing. Seven interleaved batches target 6 ms each; the report includes
all batch samples, medians, ranges and iteration counts. GC runs normally.

Component probes use prepared inputs and reusable plans/operations to isolate
work. They deliberately omit setup that complete public calls perform, so do
not add their timings together and call that a whole-call measurement. Folding
preparation includes circular operation construction and required casts. Linear
FFT execution includes padding and result materialization. CPU, compiler macros
and extension Makefile flags distinguish host information from build assumptions.

The second round adds a research-only `Prototype` subclass. It uses coefficients
frozen from the first round's training data before inspecting holdout results.
Production `lib/` and `ext/` remain unchanged. The prototype uses three measured
OS/architecture profiles for ranks 1–3; unknown platforms, scalars and higher
ranks retain the original model. All times are complete fresh invocations,
including selection, planning and casting. Prototype and original calls share
only immutable input arrays, not operation plans or prepared buffers.
An additional Ubuntu 22.04 x86 job validates the same frozen profile on an
older compiler; it supplies no coefficient-fitting data.

Run its deterministic safety checks before collecting timings:

```sh
CONVOLVER_DISABLE_SIMPLECOV=1 bundle exec rspec benchmark/recalibration/validation_spec.rb
```

The fixed tables are experimental evidence, not approved platform policy. See
the local recalibration proposal for provenance, fitting method, comparison with
a shared model, results, and remaining review decisions. A production change
needs its own implementation and validation after proposal approval.

The final comparison adds `prototype90`, changing only the FFT margin from 0.8
to 0.9. The previous validation found repeatable lost FFT wins near the 0.8
threshold on new runner CPUs. Coefficients stay frozen. Both prototype wrappers
now also mirror the production private execution-factory call, so final timing
and allocation comparisons include that dispatch layer. Earlier results did
not include this extra wrapper call; do not mix their overhead figures with the
final comparison. Both variants validate numerical results before timing.
