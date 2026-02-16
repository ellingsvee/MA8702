use std::{f64::consts::PI, ops::AddAssign, time::Instant};

use burn::{
    backend::{Autodiff, NdArray},
    tensor::{ElementConversion, backend::AutodiffBackend},
};
use mymcmc::distributions::GradientTarget;
use mymcmc::hmc::HMC;
use num_traits::Float;
use plotly::{Layout, Scatter};

struct Volcano {}

impl<T, B> GradientTarget<T, B> for Volcano
where
    T: Float + std::fmt::Debug + AddAssign + ElementConversion,
    B: AutodiffBackend,
{
    fn unnorm_logp(&self, pos: burn::Tensor<B, 1>) -> burn::Tensor<B, 1> {
        let xtx = (pos.clone() * pos.clone()).sum();
        let term1 = -xtx.clone().mul_scalar(T::from(0.5).unwrap());
        let term2 = (xtx.add_scalar(T::from(0.25).unwrap())).log();
        (term1 + term2).add_scalar(-T::from(2.0 * PI).unwrap().ln())
    }
}

fn main() {
    type Backend = Autodiff<NdArray>;

    let target = Volcano {};
    let mut sampler = HMC::<f32, Backend, Volcano>::new(target, vec![0., 0.], 0.032, 10);

    let start_time = Instant::now();
    let samples = sampler.run(10000);
    let elapsed = start_time.elapsed();
    println!("Sampling took {:.2?}", elapsed);

    let x_coords: Vec<f32> = samples.iter().map(|s| s[0]).collect();
    let y_coords: Vec<f32> = samples.iter().map(|s| s[1]).collect();

    let trace = Scatter::new(x_coords, y_coords)
        .mode(plotly::common::Mode::Markers)
        .name("MCMC samples")
        .marker(plotly::common::Marker::new().size(5));

    let layout = Layout::new()
        .title("HMC Samples from Volcano Distribution")
        .x_axis(plotly::layout::Axis::new().title("x"))
        .y_axis(plotly::layout::Axis::new().title("y"))
        .show_legend(true)
        .width(600)
        .height(600);

    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot.write_html("hmc_samples.html");
    println!("Plot saved to hmc_samples.html");
}
