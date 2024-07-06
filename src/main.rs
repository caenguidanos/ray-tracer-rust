#![feature(generic_const_exprs)]

mod assert;
mod canvas;
mod ppm;
pub use canvas::*;
use tracing::info;
mod delta;
mod exercises;
mod vectorial_space;

const STACK_SIZE: usize = 256 * 1024 * 1024;

fn main() {
    tracing_subscriber::fmt::init();

    let builder = std::thread::Builder::new()
        .name(String::from("main"))
        .stack_size(STACK_SIZE);

    info!("Stack size of thread: {}mb", STACK_SIZE / 1024 / 1024);

    builder.spawn(run).unwrap().join().unwrap();
}

fn run() {
    let t0 = std::time::Instant::now();

    exercises::chapter_2::main();

    info!("Program finished in {}ms", t0.elapsed().as_millis());
}
