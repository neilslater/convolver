# Selection benchmark

After compiling a release build, run:

```sh
bundle exec ruby -Ilib benchmark/selection.rb > selection-benchmark.json
```

The JSON records complete direct, FFT and automatic call timings, both estimates,
runtime details and sampling conditions. Inputs cover both floating types and
operations, ranks zero through five, the former 1000-element threshold, kernel
crossover cases, prime dimensions, output and boundary modes, integer conversion
and reversed views. Each case checks result dtype, shape and numerical agreement
before timing. Garbage collection runs normally; method order rotates between
batches. Do not run competing benchmarks simultaneously.

Compare automatic time with the faster explicit path in the same row. Review
median and 90th-percentile ratios and individual outliers; shared CI machines
are noisy, so timings are evidence rather than a pass/fail gate. The Ruby 4.0 CI
job uploads this report from its release build. Native lint, coverage and
sanitizer builds are unsuitable for performance calibration.
