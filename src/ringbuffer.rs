use std::i16;
use std::f32::consts::PI;

pub struct Ringbuffer {
    inner: Vec<i16>,
    pos: usize,
    window: Vec<f32>
}

impl Ringbuffer {
    pub fn new(capacity: usize) -> Ringbuffer {
        let window = (0..capacity).map(|x| 0.54 - 0.46 * (2.0*PI* (x as f32) / capacity as f32).cos()).collect();

        Ringbuffer {
            inner: vec![0; capacity],
            pos: 0,
            window
        }
    }

    pub fn append_back(&mut self, samples: &[i16]) {
        for sample in samples {
            self.inner[self.pos] = *sample;

            self.pos = (self.pos + 1) % self.inner.len();
        }
    }

    pub fn apply_hamming(&self, out: &mut [f32]){
        let mut i = 0;
        let mut j = self.pos;

        while j != self.inner.len() {
            out[i] = self.window[i] * (self.inner[j] as f32 / i16::MAX as f32);

            i += 1;
            j += 1;
        }

        j = 0;

        while j != self.pos {
            out[i] = self.window[i] * (self.inner[j] as f32 / i16::MAX as f32);

            i += 1;
            j += 1;
        }
    }
}
