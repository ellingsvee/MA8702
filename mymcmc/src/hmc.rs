use std::marker::PhantomData;

use burn::{
    Tensor,
    tensor::{Element, ElementConversion, backend::AutodiffBackend},
};
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};

use crate::distributions::GradientTarget;

pub struct HMC<T, B, GTarget>
where
    B: AutodiffBackend,
{
    pub target: GTarget,
    pub step_size: T,
    pub n_leapfrog: usize,
    pub position: Vec<T>,
    logp: T,
    grad: Vec<T>,
    pub rng: SmallRng,
    _phantom: PhantomData<B>,
}

impl<T, B, GTarget> HMC<T, B, GTarget>
where
    T: Float + Element + ElementConversion,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B>,
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
    rand_distr::StandardUniform: rand_distr::Distribution<T>,
{
    pub fn new(target: GTarget, initial_pos: Vec<T>, step_size: T, n_leapfrog: usize) -> Self {
        let rng = SmallRng::from_rng(&mut rand::rng());
        let (logp, grad) = Self::compute_logp_and_grad(&target, &initial_pos);
        Self {
            target,
            step_size,
            n_leapfrog,
            position: initial_pos,
            logp,
            grad,
            rng,
            _phantom: PhantomData,
        }
    }

    /// Compute log-probability and its gradient via Burn autodiff.
    /// This is the only place Burn tensors are used.
    fn compute_logp_and_grad(target: &GTarget, pos: &[T]) -> (T, Vec<T>) {
        let device = B::Device::default();
        let pos_tensor = Tensor::<B, 1>::from_floats(pos, &device)
            .detach()
            .require_grad();
        let logp_tensor = target.unnorm_logp(pos_tensor.clone());
        let grads = logp_tensor.backward();
        let grad_inner = pos_tensor.grad(&grads).unwrap();

        let logp_val: T = logp_tensor.into_scalar().elem();
        let grad_vec: Vec<T> = Tensor::<B, 1>::from_inner(grad_inner)
            .into_data()
            .to_vec::<T>()
            .unwrap();

        (logp_val, grad_vec)
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    pub fn run(&mut self, n_steps: usize) -> Vec<Vec<T>> {
        let dim = self.position.len();
        let mut flat: Vec<T> = Vec::with_capacity(n_steps * dim);

        for _ in 0..n_steps {
            self.step();
            flat.extend_from_slice(&self.position);
        }

        // Tensor::from_data(
        //     burn::tensor::TensorData::new(flat, [n_steps, dim]),
        //     &B::Device::default(),
        // )
        //

        // Reshape to (n_steps, dim)
        (0..n_steps)
            .map(|i| flat[i * dim..(i + 1) * dim].to_vec())
            .collect()
    }

    pub fn step(&mut self) {
        let dim = self.position.len();
        let normal = Normal::new(T::zero(), T::one()).unwrap();
        let half = T::from(0.5).unwrap();

        // Sample momentum
        let mut momentum: Vec<T> = (0..dim).map(|_| normal.sample(&mut self.rng)).collect();

        // Current Hamiltonian (plain scalar arithmetic)
        let h_current = -self.logp + ke(&momentum);

        // Leapfrog integration using plain Vec arithmetic
        let mut pos = self.position.clone();
        let mut grad = self.grad.clone();
        let half_step = self.step_size * half;
        let mut logp_prop = T::zero();

        for _ in 0..self.n_leapfrog {
            // Half-step momentum
            for (m, g) in momentum.iter_mut().zip(grad.iter()) {
                *m = *m + half_step * *g;
            }
            // Full-step position
            for (q, m) in pos.iter_mut().zip(momentum.iter()) {
                *q = *q + self.step_size * *m;
            }
            // Gradient via autodiff (the only expensive call)
            let (lp, new_grad) = Self::compute_logp_and_grad(&self.target, &pos);
            logp_prop = lp;
            grad = new_grad;
            // Half-step momentum
            for (m, g) in momentum.iter_mut().zip(grad.iter()) {
                *m = *m + half_step * *g;
            }
        }

        // Proposed Hamiltonian
        let h_proposed = -logp_prop + ke(&momentum);

        // Metropolis accept/reject
        let log_accept = h_current - h_proposed;
        let u: T = self.rng.random::<T>();

        if log_accept >= u.ln() {
            self.position = pos;
            self.logp = logp_prop;
            self.grad = grad;
        }
    }
}

#[inline]
fn ke<T: Float>(momentum: &[T]) -> T {
    let half = T::from(0.5).unwrap();
    momentum.iter().fold(T::zero(), |acc, &m| acc + m * m) * half
}
