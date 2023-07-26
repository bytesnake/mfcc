#[cfg(any(feature = "fftw", feature = "fftextern"))]
use fftw::plan::*;
#[cfg(any(feature = "fftw", feature = "fftextern"))]
use fftw::types::*;
use num_complex::Complex64;
#[cfg(feature = "fftrust")]
use rustfft::num_complex::Complex;
#[cfg(feature = "fftrust")]
use rustfft::{Fft, FftPlanner};
use std::f64;
#[cfg(feature = "fftrust")]
use std::sync::Arc;

#[cfg(feature = "fftrust")]
pub struct InverseCosineTransform {
    ifft: Arc<dyn Fft<f64>>,
    buf: Vec<Complex64>,
    buf2: Vec<Complex64>,
}

#[cfg(feature = "fftrust")]
impl InverseCosineTransform {
    pub fn new(size: usize) -> InverseCosineTransform {
        let mut planner = FftPlanner::new();
        InverseCosineTransform {
            ifft: planner.plan_fft_inverse(size),
            buf: vec![Complex::i(); size],
            buf2: vec![Complex::i(); size],
        }
    }

    pub fn transform(&mut self, input: &[f64], output: &mut [f64]) {
        let length = self.ifft.len();
        let norm = 2.0 * length as f64;

        for i in 0..length {
            let theta = i as f64 / norm * f64::consts::PI;

            self.buf[i] = Complex::from_polar(input[i] * norm.sqrt(), theta);
        }

        self.buf[0] = self.buf[0].unscale(2.0_f64.sqrt());

        //dbg!(&self.buf);
        self.ifft
            .process_with_scratch(&mut self.buf, &mut self.buf2);

        for i in 0..length / 2 {
            output[i * 2] = self.buf2[i].re / (length as f64);
            output[i * 2 + 1] = self.buf2[length - i - 1].re / (length as f64);
        }
    }
}

#[cfg(feature = "fftrust")]
pub struct ForwardRealFourier {
    fft: Arc<dyn Fft<f64>>,
    buf: Vec<Complex64>,
    buf2: Vec<Complex64>,
}

#[cfg(feature = "fftrust")]
impl ForwardRealFourier {
    pub fn new(size: usize) -> ForwardRealFourier {
        let mut planner = FftPlanner::new();
        ForwardRealFourier {
            fft: planner.plan_fft_forward(size / 2),
            buf: vec![Complex::i(); size / 2],
            buf2: vec![Complex::i(); size / 2 + 1],
        }
    }

    pub fn transform(&mut self, input: &[f64], output: &mut [Complex64]) {
        let length = self.fft.len();

        for i in 0..length {
            self.buf[i] = Complex::new(input[i * 2], input[i * 2 + 1]);
        }

        self.fft
            .process_with_scratch(&mut self.buf, &mut self.buf2[0..length]);

        let first = self.buf2[0].clone();
        self.buf2[length] = self.buf2[0];
        for i in 0..length {
            let cplx = Complex64::from_polar(
                1.0,
                -f64::consts::PI / (length as f64) * (i as f64) + f64::consts::PI / 2.0,
            );
            output[i] = 0.5 * (self.buf2[i] + self.buf2[length - i].conj())
                + 0.5 * cplx * (-self.buf2[i] + self.buf2[length - i].conj());
        }

        output[length] = Complex64::new(first.re - first.im, 0.0);
    }
}

#[cfg(any(feature = "fftw", feature = "fftextern"))]
pub struct InverseCosineTransform {
    dct_state: R2RPlan64,
}

#[cfg(any(feature = "fftw", feature = "fftextern"))]
impl InverseCosineTransform {
    pub fn new(size: usize) -> InverseCosineTransform {
        InverseCosineTransform {
            dct_state: R2RPlan::aligned(&[size], R2RKind::FFTW_REDFT01, Flag::ESTIMATE).unwrap(),
        }
    }

    pub fn transform(&mut self, input: &mut [f64], output: &mut [f64]) {
        input[0] *= 2.0f64.sqrt();

        self.dct_state.r2r(input, output).unwrap();

        for x in output {
            *x = *x / (2.0 * input.len() as f64).sqrt();
        }
    }
}

#[cfg(any(feature = "fftw", feature = "fftextern"))]
pub struct ForwardRealFourier {
    fft_state: R2CPlan64,
}

#[cfg(any(feature = "fftw", feature = "fftextern"))]
impl ForwardRealFourier {
    pub fn new(size: usize) -> ForwardRealFourier {
        ForwardRealFourier {
            fft_state: R2CPlan::aligned(&[size], Flag::ESTIMATE).unwrap(),
        }
    }

    pub fn transform(&mut self, input: &mut [f64], output: &mut [Complex64]) {
        self.fft_state.r2c(input, output).unwrap();
    }
}
