use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mfcc::Transform;

pub fn calculate_mfcc(c: &mut Criterion) {
    let input = (0..1024)
        .map(|x| (32000.0 * (x as f32 / 1024.0 * 20.0 * 3.1415).sin()) as i16)
        .collect::<Vec<i16>>();
    let mut output = vec![0.0; 60];

    let mut mfcc = Transform::new(16000, 1024).nfilters(20, 40).normlength(10);

    c.bench_function("transform", |b| {
        b.iter(|| mfcc.transform(black_box(&input), black_box(&mut output)))
    });
}

criterion_group!(benches, calculate_mfcc,);
criterion_main!(benches);
