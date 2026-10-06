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
