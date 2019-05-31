# Mel Frequency Cepstral Coefficients

A common pre-processing step in Machine Learning with audio signals is the application of a _Mel Frequency Cepstral Coefficients_ (MFCC) transformation. They compress the signal to a very small number of coefficients (around 16 for every 10ms) and decorrelates the signal to express only the transmission function (e.g. only the formants of a utterance not the pitch). This makes them very popular in Automatic Speech Recognition (ASR), Room Classification, Speaker Recognition etc. 

## Usage
Add this to your `Cargo.toml`
``` toml
[dependencies]
mfcc = "0.1"
```

The library can use two different FFT libraries. Either use `rustfft` (a pure rust FFT implementation) with the standard feature _fftrust_ or use `fftw` (a popular FFT library) with
```toml
[dependencies.mfcc]
version = "0.1"
default-features = false
features = ["fftextern"]
```

A rough benchmark shows that their performance are comparable, for _FFTW_:
```
test tests::bench_mfcc ... bench:     123,959 ns/iter (+/- 22,979)
```
For _rustfft_:
```
test tests::bench_mfcc ... bench:     162,603 ns/iter (+/- 35,914)
```

## How it works

First you need to segment you audio data in chunks of around 10ms-20ms (max 1024 samples for 48kHz). From these you can calculate the MFCC coefficients with
```rust
use mfcc::Transform;

let mut state = Transform::new(48000, 1024)
    .nfilters(20, 40)
    .normlength(10);

let mut output = vec![0.0; 20*3];

state.transform(&input, &mut output);
```

This creates MFCCs for an input of 1024 samples of 48000kHz sample rate. They are converted with 40 Mel Filter banks into the cepstral domain and finally the first 20 of them are written to the output. Then first and second order derivatives are computed and they are normalized over a range of 10 sets (max for energy/mean for the rest).

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
