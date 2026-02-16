use ndarray::LinalgScalar;
use ndarray::{Array2, ArrayView1};

pub trait MarkovChain<T> {
    fn step(&mut self);
    fn state(&self) -> &[T];
    fn dim(&self) -> usize {
        self.state().len()
    }
}

pub fn run_chain<M, T>(chain: &mut M, n_steps: usize) -> Array2<T>
where
    M: MarkovChain<T>,
    T: LinalgScalar,
{
    let dim = chain.dim();
    let mut out = Array2::<T>::zeros((n_steps, dim));

    for i in 0..n_steps {
        chain.step();
        let state = chain.state();
        let view = ArrayView1::from(state);
        out.row_mut(i).assign(&view);
    }
    out
}
