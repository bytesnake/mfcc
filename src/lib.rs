#![feature(core_intrinsics)]
#![feature(test)]

extern crate test;

mod freqs;
pub mod mfcc;
mod ringbuffer;

#[cfg(test)]
mod tests {
    use test::Bencher;
    use crate::mfcc::Transform;


    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[bench]
    fn bench_mfcc(b: &mut Bencher) {
        let input = (0..1024).map(|x| ( 32000.0 * (x as f32 / 1024.0 * 20.0 * 3.1415).sin()) as i16).collect::<Vec<i16>>();
        let mut output = vec![0.0; 48];

        let mut mfcc = Transform::new(16000, 1024);

        b.iter(|| mfcc.transform(&input, &mut output));
    }
}
