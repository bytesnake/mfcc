use std::f64;
use std::sync::Arc;
use rustfft::{FFT, FFTplanner};
use rustfft::num_complex::{Complex, Complex64};

pub struct InverseCosineTransform {
    ifft: Arc<dyn FFT<f64>>,
    buf: Vec<Complex64>,
    buf2: Vec<Complex64>
}

impl InverseCosineTransform {
    pub fn new(size: usize) -> InverseCosineTransform {
        let mut planner = FFTplanner::new(false);
        InverseCosineTransform {
            ifft: planner.plan_fft(size),
            buf: vec![Complex::i(); size],
            buf2: vec![Complex::i(); size]
        }
    }

    pub fn transform(&mut self, input: &[f64], output: &mut [f64]) {
        let length = self.ifft.len();
        let norm = 2.0 * length as f64;

        for i in 0..length {
            let theta = i as f64 / norm * f64::consts::PI;

            self.buf[i] = Complex::from_polar(&(input[i] * norm.sqrt()), &theta);
        }

        self.buf[0] = self.buf[0].unscale(2.0_f64.sqrt());

        self.ifft.process(&mut self.buf, &mut self.buf2);

        for i in 0..length {
            output[i*2] = self.buf2[i].re;
            output[i*2+1] = self.buf2[length - i - 1].re;
        }
    }
}

pub struct ForwardRealFourier {
    fft: Arc<dyn FFT<f64>>,
    buf: Vec<Complex64>,
    buf2: Vec<Complex64>
}

impl ForwardRealFourier {
    pub fn new(size: usize) -> ForwardRealFourier {
        let mut planner = FFTplanner::new(false);
        ForwardRealFourier {
            fft: planner.plan_fft(size/2),
            buf: vec![Complex::i(); size/2],
            buf2: vec![Complex::i(); size/2+1]
        }
    }

    pub fn transform(&mut self, input: &[f64], output: &mut [Complex64]) {
        let length = self.fft.len();

        for i in 0..length {
            self.buf[i] = Complex::new(input[i*2], input[i*2 + 1]);
        }

        self.fft.process(&mut self.buf, &mut self.buf2[0..length]);

        let first = self.buf2[0].clone();
        self.buf2[length] = self.buf2[0];

        for i in 0..length {
            let phase = -2.0 * f64::consts::PI / (length as f64) * (i as f64) + f64::consts::PI / 2.0;

            let part1 = self.buf2[i] + self.buf2[length - i].conj();
            let part2 = self.buf2[i] - self.buf2[length - i].conj();
            
            output[length - i] = 0.5 * (part1 - part2 * Complex64::from_polar(&1.0, &phase));
        }

        //output[0] = Complex64::new(f, 0.0);
        output[0] = Complex64::new(output[length].re - output[length].im, 0.0);
    }
}

