use std::f64::consts::PI;
use std::i16;

pub struct Ringbuffer {
    inner: Vec<i16>,
    pos: usize,
    window: Vec<f64>,
}

impl Ringbuffer {
    pub fn new(capacity: usize) -> Ringbuffer {
        let window = (0..capacity)
            .map(|x| 0.54 - 0.46 * (2.0 * PI * (x as f64) / capacity as f64).cos())
            .collect();

        Ringbuffer {
            inner: vec![0; capacity],
            pos: 0,
            window,
        }
    }

    pub fn append_back(&mut self, samples: &[i16]) {
        for sample in samples {
            self.inner[self.pos] = *sample;

            self.pos = (self.pos + 1) % self.inner.len();
        }
    }

    pub fn apply_hamming(&self, out: &mut [f64]) {
        let mut i = 0;
        let mut j = self.pos;

        while j != self.inner.len() {
            out[i] = self.window[i] * (self.inner[j] as f64 / i16::MAX as f64);

            i += 1;
            j += 1;
        }

        j = 0;

        while j != self.pos {
            out[i] = self.window[i] * (self.inner[j] as f64 / i16::MAX as f64);

            i += 1;
            j += 1;
        }
    }
}
