# frozen_string_literal: true

module SelectionBenchmark
  # Median batch timings with interleaved methods and normal garbage collection.
  class Measurement
    def initialize(calls)
      @calls = calls
      @counts = calls.transform_values { |call| iterations(call) }
      @times = calls.transform_values { [] }
    end

    def run
      5.times do |batch|
        @calls.keys.rotate(batch).each { |key| @times[key] << (elapsed(@calls[key], @counts[key]) * 1e6) }
      end
      { median_us: @times.transform_values { |values| values.sort[2] },
        range_us: @times.transform_values(&:minmax), iterations: @counts }
    end

    private

    def iterations(call)
      3.times { call.call }
      (0.004 / elapsed(call, 3)).ceil.clamp(3, 200)
    end

    def elapsed(call, count)
      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      count.times { call.call }
      (Process.clock_gettime(Process::CLOCK_MONOTONIC) - start) / count
    end
  end
end
