# frozen_string_literal: true

module SelectionBenchmark
  # Median batch timings with interleaved methods and normal garbage collection.
  class Measurement
    def initialize(calls, samples: 5, duration: 0.004, maximum: 200)
      @calls = calls
      @samples = samples
      @duration = duration
      @maximum = maximum
      @counts = calls.transform_values { |call| iterations(call) }
      @times = calls.transform_values { [] }
    end

    def run
      @samples.times do |batch|
        @calls.keys.rotate(batch).each { |key| @times[key] << (elapsed(@calls[key], @counts[key]) * 1e6) }
      end
      { median_us: @times.transform_values { |values| values.sort[@samples / 2] },
        range_us: @times.transform_values(&:minmax), iterations: @counts, samples_us: @times }
    end

    private

    def iterations(call)
      3.times { call.call }
      (@duration / elapsed(call, 3)).clamp(3, @maximum).ceil
    end

    def elapsed(call, count)
      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      count.times { call.call }
      (Process.clock_gettime(Process::CLOCK_MONOTONIC) - start) / count
    end
  end
end
