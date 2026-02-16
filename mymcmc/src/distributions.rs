use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use num_traits::Float;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

pub trait Proposal<T, F: Float> {
    fn sample(&mut self, current: &[T]) -> Vec<T>;
    fn logp(&self, current: &[T], proposed: &[T]) -> F;
    fn set_seed(&mut self, seed: u64);
}

pub trait Target<T, F: Float> {
    fn unnorm_logp(&self, pos: &[T]) -> F;
}

pub trait Normalized<T, F: Float> {
    fn logp(&self, pos: &[T]) -> F;
}

pub trait GradientTarget<T: Float, B: AutodiffBackend> {
    fn unnorm_logp(&self, pos: Tensor<B, 1>) -> Tensor<B, 1>;

    fn unnorm_logp_and_grad(&self, pos: Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let pos = pos.clone().detach().require_grad();
        let unnorm_logp = self.unnorm_logp(pos.clone());
        let unnorm_logp_grad_inner = pos.grad(&unnorm_logp.backward()).unwrap();
        let unnorm_logp_grad = Tensor::<B, 1>::from_inner(unnorm_logp_grad_inner);
        (unnorm_logp, unnorm_logp_grad)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DifferentiableGaussian2D<T: Float> {
    pub mean: [T; 2],
    pub cov: [[T; 2]; 2],
    pub inv_cov: [[T; 2]; 2],
    pub logdet_cov: T,
    pub norm_const: T,
}

impl<T> DifferentiableGaussian2D<T>
where
    T: Float + std::fmt::Debug + num_traits::FloatConst,
{
    pub fn new(mean: [T; 2], cov: [[T; 2]; 2]) -> Self {
        let det_cov = cov[0][0] * cov[1][1] - cov[1][0] * cov[0][1];
        let inv_det = T::one() / det_cov;
        let inv_cov = [
            [cov[1][1] * inv_det, -cov[0][1] * inv_det],
            [-cov[1][0] * inv_det, cov[0][0] * inv_det],
        ];
        let logdet_cov = det_cov.ln();
        let two = T::one() + T::one();
        let norm_const = -(two * (two * T::PI()).ln() + logdet_cov) / two;
        Self {
            mean,
            cov,
            inv_cov,
            logdet_cov,
            norm_const,
        }
    }
}

impl<T, B> GradientTarget<T, B> for DifferentiableGaussian2D<T>
where
    T: Float + burn::tensor::ElementConversion + core::fmt::Debug + burn::tensor::Element,
    B: AutodiffBackend,
{
    fn unnorm_logp(&self, pos: Tensor<B, 1>) -> Tensor<B, 1> {
        let dims = pos.dims()[0];
        assert_eq!(dims, 2, "Expected dimension=2");

        let mean_tensor =
            Tensor::<B, 1>::from_floats([self.mean[0], self.mean[1]], &B::Device::default());

        let delta = pos - mean_tensor;

        let inv_cov_data = [
            [self.inv_cov[0][0], self.inv_cov[0][1]],
            [self.inv_cov[1][0], self.inv_cov[1][1]],
        ];
        let inv_cov_tensor = Tensor::<B, 2>::from_floats(inv_cov_data, &B::Device::default());

        let z = delta.clone().reshape([1, 2]).matmul(inv_cov_tensor);
        let quad = (z.reshape([2]) * delta).sum();
        let half = T::from(2).unwrap();

        -quad.mul_scalar(half) + self.norm_const
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsotropicGaussian<T: Float> {
    pub std: T,
    rng: SmallRng,
}

impl<T: Float> IsotropicGaussian<T> {
    pub fn new(std: T) -> Self {
        Self {
            std,
            rng: SmallRng::from_rng(&mut rand::rng()),
        }
    }
}

impl<T: Float + std::ops::AddAssign> Proposal<T, T> for IsotropicGaussian<T>
where
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
{
    fn sample(&mut self, current: &[T]) -> Vec<T> {
        let normal = Normal::new(T::zero(), self.std).unwrap();

        normal
            .sample_iter(&mut self.rng)
            .zip(current)
            .map(|(eps, x)| eps + *x)
            .collect()
    }

    fn logp(&self, current: &[T], proposed: &[T]) -> T {
        let mut logp = T::zero();
        let d = T::from(current.len()).unwrap();
        let two = T::from(2).unwrap();
        let var = self.std * self.std;
        for (&c, &p) in current.iter().zip(proposed) {
            let diff = p - c;
            let exponent = -(diff * diff) / (two * var);
            logp += exponent
        }
        logp +=
            -d * T::from(0.5).unwrap() * (var * T::from(PI).unwrap() * self.std * self.std).ln();
        logp
    }

    fn set_seed(&mut self, seed: u64) {
        self.rng = SmallRng::seed_from_u64(seed);
    }
}

impl<T: Float + std::ops::AddAssign> Target<T, T> for IsotropicGaussian<T> {
    fn unnorm_logp(&self, pos: &[T]) -> T {
        let mut sum = T::zero();
        for &x in pos.iter() {
            sum += x * x;
        }
        -T::from(0.5).unwrap() * sum / (self.std * self.std)
    }
}
