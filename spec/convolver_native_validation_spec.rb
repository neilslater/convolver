# frozen_string_literal: true

require 'helpers'

describe Convolver do
  # Public methods reject these inputs in Ruby before the native boundary.
  # Call each private primitive directly to exercise its independent checks.
  %i[convolve_basic_valid correlate_basic_valid].each do |method_name|
    describe ".#{method_name}" do
      it 'rejects a signal that is not a Numo array' do
        expect { described_class.send(method_name, [1.0], NArray[1.0]) }
          .to raise_error(ArgumentError, 'signal and kernel must be Numo::NArray values')
      end

      it 'rejects a kernel that is not a Numo array' do
        expect { described_class.send(method_name, NArray[1.0], [1.0]) }
          .to raise_error(ArgumentError, 'signal and kernel must be Numo::NArray values')
      end

      it 'rejects an empty signal' do
        expect { described_class.send(method_name, NArray.zeros(0), NArray[1.0]) }
          .to raise_error(ArgumentError, 'signal and kernel must not be empty')
      end

      it 'rejects an empty kernel' do
        expect { described_class.send(method_name, NArray[1.0], NArray.zeros(0)) }
          .to raise_error(ArgumentError, 'signal and kernel must not be empty')
      end

      it 'rejects a signal with fewer dimensions than the kernel' do
        expect { described_class.send(method_name, NArray.ones(2), NArray.ones(2, 2)) }
          .to raise_error(ArgumentError, 'signal and kernel must have equal rank')
      end

      it 'rejects a signal with more dimensions than the kernel' do
        expect { described_class.send(method_name, NArray.ones(2, 2), NArray.ones(2)) }
          .to raise_error(ArgumentError, 'signal and kernel must have equal rank')
      end

      it 'rejects equal ranks above the native limit without large allocations' do
        rank_17_array = NArray.ones(*([1] * 17))

        expect { described_class.send(method_name, rank_17_array, rank_17_array) }
          .to raise_error(ArgumentError, 'maximum supported rank is 16')
      end

      it 'rejects a kernel larger in the first signal dimension' do
        expect { described_class.send(method_name, NArray.ones(1, 2), NArray.ones(2, 1)) }
          .to raise_error(ArgumentError, 'kernel must not be larger than signal in any dimension')
      end

      it 'rejects a kernel larger in the final signal dimension' do
        expect { described_class.send(method_name, NArray.ones(2, 1), NArray.ones(1, 2)) }
          .to raise_error(ArgumentError, 'kernel must not be larger than signal in any dimension')
      end
    end
  end
end
