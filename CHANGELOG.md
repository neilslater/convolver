# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `:valid`, `:same`, and `:full` output modes to every calculation and
  estimator method.
- Added constant, nearest, reflect, mirror, and wrap signal boundary extensions,
  including nonzero constant fill and kernels larger than the signal where the
  output mode permits them.
- Added scalar and per-axis kernel origins with an explicit even-kernel
  alignment convention.
- Added an origin-aware circular PocketFFT correlation path for periodic
  same-sized output.

### Changed

- Made algorithm selection and both public time estimators account for output
  shape, boundary extension, kernel folding, and FFT working shape.
- Moved the native valid-only calculation behind a shared Ruby orchestration
  layer so direct and FFT implementations use identical option validation and
  boundary semantics.

## [1.0.1] - 2026-07-29

### Fixed

- Made the direct native implementation handle non-contiguous Numo::NArray
  views safely.
- Prevented an out-of-bounds native buffer write for rank-16 inputs.
- Replaced unchecked native size and offset arithmetic with overflow-checked
  calculations.

## [1.0.0] - 2026-07-21

### Added

- Native C quality tasks for strict compiler warnings (`c:lint`), GCC/gcovr
  coverage (`c:coverage`), and ASan/UBSan checks (`c:sanitize`).
- GitHub Actions reporting and downloadable HTML and Cobertura artifacts for C
  coverage.
- RuboCop checks with Rake and RSpec plugins.
- SimpleCov reporting for the Ruby test suite.
- Support for Ruby 4.0 in the CI test matrix.
- Monthly scheduled CI runs to detect dependency and toolchain regressions.
- Ruby branch-coverage reporting and additional coverage of native input
  validation.

### Changed

- **Breaking:** Replaced the unmaintained `narray` dependency with
  `numo-narray-alt`. Public methods now accept `Numo::NArray` values and return
  `Numo::SFloat` results instead of legacy `NArray` values.
- **Breaking:** Raised the minimum supported Ruby version to 3.2.
- Replaced the `fftw3` gem and external FFTW library with `numo-pocketfft`,
  removing the system FFTW installation requirement.
- Renamed the FFT-backed implementation from `convolve_fftw3` to
  `convolve_fft`. The old name remains as a deprecated compatibility alias.
- Clarified that the operations calculate valid cross-correlation and expanded
  the README and YARD documentation for input, output, and shape behaviour.
- Replaced the retired Travis CI configuration with GitHub Actions.
- Improved gem metadata and limited packaged files to the library, native
  extension, documentation, and licence.

### Fixed

- Reject non-Numo inputs, empty arrays, unequal ranks, kernels larger than their
  signals, and ranks unsupported by the native implementation with descriptive
  `ArgumentError` messages.
- Updated the native extension to use the maintained Numo C API and removed the
  legacy untyped-data compatibility code.

[Unreleased]: https://github.com/neilslater/convolver/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/neilslater/convolver/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/neilslater/convolver/compare/v0.3.2...v1.0.0
