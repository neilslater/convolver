# Convolver

Adds a fast convolve operation to NArray.

## Installation

Add this line to your application's Gemfile:

    gem 'convolver'

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install convolver

## Usage

Basic convolution:

   a = NArray[0.3,0.4,0.5]
   b = NArray[1.3, -0.5]
   c = a.convolve( b )
   => NArray.float(2): [ 0.19, 0.27 ]

## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request
