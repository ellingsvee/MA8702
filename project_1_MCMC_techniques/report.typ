#import "template.typ": *
#import "@preview/cetz:0.4.2"
#import "@preview/muchpdf:0.1.2": muchpdf

// Algorithm
#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

#let py(body) = raw(body, lang: "python")
#let cmd(body) = raw(body, lang: "bash")

#set math.equation(numbering: "(1)")
#show math.equation: it => {
  if it.block and not it.has("label") and it.numbering != none [
    #counter(math.equation).update(v => v - 1)
    #math.equation(it.body, block: true, numbering: none)
  ] else {
    it
  }
}


#show: ilm.with(
  title: [Metropolis-Hastings for bivariate densities],
  author: "Project 1 in MA8702. Written by Elling Svee.",
  // date: datetime(year: 2026, month: 01, day: 18),
  date: datetime.today(),
  // abstract: [],
  bibliography: bibliography("refs.bib", style: "elsevier-harvard"),
  figure-index: (enabled: false),
  table-index: (enabled: false),
  listing-index: (enabled: false),
  table-of-contents: none,
  chapter-pagebreak: false,
  fancy-cover-page: true,
)

= Example densities

In this project we implement three variations of the _Metropolis-Hastings_ (MH). For evaluating the algorithms we consider three bivariate target densities for $bold(x) = (x, y)^top$:

+ *Gaussian distribution:* The first is a bivariate Gaussian distribution with correlation. Its probability density function (PDF) is
  $
    pi(bold(x)) = frac(1, 2 pi det(bold(upright(Sigma)))^(1/2)) exp lr((-frac(1, 2)bold(x)^top bold(upright(Sigma))^(-1)bold(x)))
  $
  where $bold(upright(Sigma))$ has $1$ on the diagonal and $0.9$ on the off diagonals.

+ *Multimodal density:* The second is a multimodal density constructed as a mixture of Gaussian densities. Its PDF is
  $
    pi(bold(x)) = sum_(i=1)^(3) w_i frac(1, 2 pi det(bold(upright(Sigma))_i)^(1/2)) exp lr((-frac(1, 2)(bold(x)-bold(mu)_i)^top bold(upright(Sigma))_(i)^(-1)(bold(x) - bold(mu)_i))),
  $
  with $w_i = 1 \/3$ for $i=1, 2, 3$. The means are $bold(mu)_1 = (−1.5, −1.5)^(top)$, $bold(mu)_2 = (1.5, 1.5)^(top)$ and $bold(mu)_3 = (-2, 2)^(top)$, and the covariance matrices all have correlation $0$ and variances $sigma_(1)^(2) = sigma_(2)^(2) = 1$ and $sigma_3^(2) = 0.8$.

+ *Volcano density:* Lastly we consider a volcano-shaped density with PDF
  $
    pi(bold(x)) prop frac(1, 2 pi) exp lr((-1/2 bold(x)^(top) bold(x)))(bold(x)^(top) bold(x) + 0.25)
  $


The densities are implemented in the `densities.py` file. @bivariate-densities visualizes the three densities on a grid covering $[-5, 5] times [-5, 5]$. The grid spacing is $0.1$, which gives in total $101 times 101$ grid cells.

#figure(
  image("code/output/distributions.svg", width: 100%),
  caption: [Bivariate densities on a $[-5, 5] times [-5, 5]$ domain.],
)<bivariate-densities>

// The densities are implemented in the `densities.py` file, which contains the following code.
//
// #raw(read("code/scripts/densities.py"), block: true, lang: "python")

= Theoretical background for Metropolis-Hastings

The MH algorithm is a _Markov chain Monte Carlo_ (MCMC) method for sampling from a target distribution $pi(bold(x))$ known only up to a normalizing constant. Given a current state $bold(x)^((t))$, the algorithm proceeds as follows:

+ *Propose* a candidate $bold(x)'$ from a proposal distribution $q(bold(x)'|bold(x)^((t)))$.
+ *Compute* the acceptance probability
  $
    alpha(bold(x)^((t)), bold(x)') = min lr((1, frac(pi(bold(x)') q(bold(x)^((t))|bold(x)'), pi(bold(x)^((t))) q(bold(x)'|bold(x)^((t)))))).
  $
+ *Accept* the proposal with probability $alpha$: set $bold(x)^((t+1)) = bold(x)'$. Otherwise, set $bold(x)^((t+1)) = bold(x)^((t))$.

The acceptance ratio ensures the chain satisfies _detailed balance_ with respect to $pi$:
$
  pi(bold(x)) P(bold(x) -> bold(x)') = pi(bold(x)') P(bold(x)' -> bold(x)),
$
which guarantees that $pi$ is a stationary distribution of the chain.

= Code setup

The three MCMC algorithms are implemented in Python using the JAX @jax2018github library for linear algebra and _automatic differentiation_ (AD). This enables us to compute the gradients of the log-densities required for the Langevin and Hamiltonian algorithms without having to specify them manually. JAX also has functionality for just-in-time (JIT) compilation, which we use to speed up the sampling process, and for running multiple chains in parallel. The implementation is overall inspired by the great Blackjax library @cabezas2024blackjax, which provides modern and efficient implementations of many MCMC algorithms.


When developing the code, I found it most sensible to implement the three algorithms as separate _kernels_, and write a single function for that takes a specific kernel as input and runs the chains. This inference function is then used for all three algorithms, and allows for a general and reusable implementation. The code can be found in `scripts/inference.py`.

// #raw(read("code/scripts/inference.py"), block: true, lang: "python")




// / `densities.py`: Implementation of the three densities.
//
// #raw(read("code/scripts/densities.py"), block: true, lang: "python")
//
// / `inderence.py`: For running the chains.
//
// #raw(read("code/scripts/inference.py"), block: true, lang: "python")
//
// / `autocorr.py`: Computing the autocorrelations.
//
// #raw(read("code/scripts/autocorr.py"), block: true, lang: "python")
//
// / `tuning_experiment.py`: Running the algorithms for different $sigma$-values, and plotting results.
//
// #raw(read("code/scripts/tuning_experiment.py"), block: true, lang: "python")


= Random-walk Metropolis-Hastings

== Theory

Random-walk MH uses a symmetric proposal centered at the current state:
$
  q(bold(x)'|bold(x)) = cal(N)(bold(x)' ; bold(x), sigma^2 bold(I)),
$
where $sigma > 0$ is the step size. Since the proposal is symmetric, i.e., $q(bold(x)'|bold(x)) = q(bold(x)|bold(x)')$, the acceptance probability simplifies to
$
  alpha(bold(x), bold(x)') = min lr((1, frac(pi(bold(x)'), pi(bold(x))))).
$

The step size $sigma$ controls the trade-off between exploration and acceptance rate. A small $sigma$ yields high acceptance but slow exploration; a large $sigma$ proposes distant points but with low acceptance. For high-dimensional targets, optimal scaling theory @roberts_rosenthal_2001 suggests tuning $sigma$ to achieve an acceptance rate of approximately $0.234$.


Building the Random-walk MH kernel is done in `scripts/random_walk.py`. I use the tuples `RWState` for storing the intermediate states of the chain, and `RWInfo` for storing information about the acceptance rate and other diagnostics. The `build_kernel()` function takes in the log-density function and a step size, and returns a kernel function that takes in a state and returns the next state and info. As discussed previously, this function is passed into the `inference_loop()` function that runs the chain.


// #raw(read("code/scripts/random_walk.py"), block: true, lang: "python")

// / `rwmh_chain.py`: Running the tuning experiment for the Random-walk MH.
// #raw(read("code/tasks/rwmh_chain.py"), block: true, lang: "python")

== Results
=== Gaussian distribution

@rwmh-guassian shows the tuning experiment for the Gaussian distribution. Observe that $sigma = 1.5$ gives an acceptance rate of $0.264$. This is closest to the theoretical optimal, and is therefore preferred.

#figure(
  image("code/output/rwmh/tuning_mvn.svg", width: 100%),
  caption: "Tuning experiment of Random-walk MH for Gaussian distribution",
)<rwmh-guassian>

=== Multimodal distribution

@rwmh-multimodal shows the tuning experiment for the multimodal distribution. $sigma = 1.5$ gives an acceptance rate of $0.503$. This is closest to the theoretical optimal, and is therefore preferred.

#figure(
  image("code/output/rwmh/tuning_multimodal.svg", width: 100%),
  caption: "Tuning experiment of Random-walk MH for multimodal distribution",
)<rwmh-multimodal>

=== Volcano distribution

@rwmh-volcano shows the tuning experiment for the volcano distribution. $sigma = 1.5$ gives an acceptance rate of $0.523$. This is closest to the theoretical optimal, and is therefore preferred.

#figure(
  image("code/output/rwmh/tuning_volcano.svg", width: 100%),
  caption: "Tuning experiment of Random-walk MH for volcano distribution",
)<rwmh-volcano>



= Langevin Metropolis-Hastings

== Theory

The _Metropolis-adjusted Langevin algorithm_ (MALA) incorporates gradient information into the proposal. It is motivated by the Langevin diffusion, the stochastic differential equation
$
  dif bold(X)_t = nabla log pi(bold(X)_t) dif t + sqrt(2) dif bold(W)_t,
$
which has $pi$ as its stationary distribution. Discretizing with step size $epsilon$ yields the proposal
$
  q(bold(x)'|bold(x)) = cal(N)lr((bold(x)'; bold(x) + epsilon nabla log pi(bold(x)), 2 epsilon bold(I))).
$
Unlike the random-walk proposal,  $q(bold(x)'|bold(x)) != q(bold(x)|bold(x)')$ mean this is _not_ symmetric. Therefore, we must use the full MH acceptance probability
$
  alpha(bold(x), bold(x)') = min lr((1, frac(pi(bold(x)') q(bold(x)|bold(x)'), pi(bold(x)) q(bold(x)'|bold(x))))).
$
The gradient mean proposal move toward high-density regions, enabling larger step sizes and faster mixing compared to random-walk MH. The scaling limit literature indicates that the optimal acceptance probability is approximately $0.57$ @dunson_hastings_2020.

// == Implementation
For the implementation, the `build_kernel()` in `langevin_chain.py` is structured similarly as for the random-walk MH. One thing to note is the `jax.value_and_grad()`, which computes both the value and the gradient of the log-density in a single pass.
// #raw(read("code/scripts/langevin.py"), block: true, lang: "python")

== Results
=== Gaussian distribution

@langevin-guassian shows the tuning experiment for the Gaussian distribution. $sigma=0.5$ is closest to the optimal acceptance probability, and is therefore preferred.

#figure(
  image("code/output/langevin/tuning_mvn.svg", width: 100%),
  caption: "Tuning experiment of Langevin MH for Gaussian distribution",
)<langevin-guassian>

=== Multimodal distribution

@langevin-multmodal shows the tuning experiment for the multimodal distribution. $sigma=1.0$ is closest to the optimal acceptance probability, and is therefore preferred.

#figure(
  image("code/output/langevin/tuning_multimodal.svg", width: 100%),
  caption: "Tuning experiment of Langevin MH for multimodal distribution",
)<langevin-multmodal>

=== Volcano distribution

@langevin-volcano shows the tuning experiment for the volcano distribution. $sigma=1.5$ is closest to the optimal acceptance probability, and is therefore preferred.

#figure(
  image("code/output/langevin/tuning_volcano.svg", width: 100%),
  caption: "Tuning experiment of Langevin MH for multimodal distribution",
)<langevin-volcano>

= Hamiltonian Metropolis-Hastings

== Theory

_Hamiltonian Monte Carlo_ (HMC) augments the target with auxiliary _momentum_ variables $bold(p) in RR^d$ and samples from the joint distribution
$
  pi(bold(x), bold(p)) prop pi(bold(x)) exp lr((-frac(1, 2) bold(p)^top bold(p))).
$
This defines a Hamiltonian system with potential energy $U(bold(x)) = -log pi(bold(x))$ and kinetic energy $K(bold(p)) = frac(1, 2) bold(p)^top bold(p)$, giving total energy (Hamiltonian)
$
  H(bold(x), bold(p)) = U(bold(x)) + K(bold(p)).
$

Each iteration proceeds as follows:
+ *Resample momentum*: Draw $bold(p) tilde cal(N)(bold(0), bold(I))$ independently of $bold(x)$.
+ *Simulate dynamics*: Integrate Hamilton's equations for $L$ steps using the leapfrog integrator with step size $epsilon$:
  $
    bold(p)_(t + epsilon\/2) & = bold(p)_t + frac(epsilon, 2) nabla log pi(bold(x)_t) \
       bold(x)_(t + epsilon) & = bold(x)_t + epsilon bold(p)_(t + epsilon\/2) \
       bold(p)_(t + epsilon) & = bold(p)_(t + epsilon\/2) + frac(epsilon, 2) nabla log pi(bold(x)_(t + epsilon))
  $
+ *Accept/reject*: Accept the proposal $(bold(x)', bold(p)')$ with probability $min(1, exp(-Delta H))$, where $Delta H = H(bold(x)', bold(p)') - H(bold(x), bold(p))$.

The leapfrog integrator is _symplectic_ (volume-preserving and time-reversible), which ensures the proposal mechanism is symmetric. In exact arithmetic $Delta H = 0$. However, in practice, small discretization errors require the MH correction. HMC can traverse the state space rapidly by following the geometry of $pi$, achieving low autocorrelation even with high acceptance rates.

As for the other algorithms, the `build_kernel()` in `scripts/hmc_chain.py` returns a kernel function that updates the state and info. Like in the MALA implementation, it also relies AD to compute the gradients. The leapfrog integrator is implemented in the `leapfrog()` function, which is called from inside the kernel.

// #raw(read("code/scripts/hamiltonian.py"), block: true, lang: "python")

// / `hmc_chain.py`: Running the tuning experiment for the HMC.
// #raw(read("code/tasks/hmc_chain.py"), block: true, lang: "python")

== Gaussian distribution

@hmc-guassian shows the tuning experiment for the Gaussian distribution. For Hamiltonian MC we prefer a high acceptance rate. Looking at the plots we see that $sigma=0.5$ gives the smallest correlation between consecutive samples. As it still has a very high acceptance rate, this is preferred. Compared to the other algorithms, we also see that HMC can achieve much lower autocorrelation even with a high acceptance rate, which is a key advantage of this method.

#figure(
  image("code/output/hmc/tuning_mvn.svg", width: 100%),
  caption: "Tuning experiment of HMC for Gaussian distribution",
)<hmc-guassian>

== Multimodal distribution

@hmc-multmodal shows the tuning experiment for the multimodal distribution. Again, the $sigma=0.5$ gives fast-decreasing automatic while retaining a high acceptance rate. This is therefore preferred.

#figure(
  image("code/output/hmc/tuning_multimodal.svg", width: 100%),
  caption: "Tuning experiment of HMC for multimodal distribution",
)<hmc-multmodal>

== Volcano distribution

@hmc-volcano shows the tuning experiment for the volcano distribution. For the same reasons as for the previous other distributions, the $sigma = 0.5$ is preferred.

#figure(
  image("code/output/hmc/tuning_volcano.svg", width: 100%),
  caption: "Tuning experiment of HMC for volcano distribution",
)<hmc-volcano>

= Stan

For the last part of the project, we implement a simple model in the probabilistic programming language Stan @carpenter2017stan. This language implements a variant of HMC called the _No-U-Turn Sampler_ (NUTS) @hoffman_no-u-turn_2011, which automatically tunes the number of leapfrog steps and step size during sampling. In my personal opinion, developing and using a separate language for MCMC seems like bit of an overkill. However, I am open to giving it a try and see how it works in practice.

We consider an example concerning the number of failures in ten power plants. The data is presented in @data_power_plants. Here $y_i$ is the number of times that pump $i$ has failed and $t_i$ is the operation time for the pump (in 1000s of hours). Pump failures are modelled as
$
  y_i|lambda_i tilde.op "Posson"(lambda_(i)t_(i)), quad i = 1, ..., 10.
$
We chose a conjugate prior for $lambda_i$
$
  lambda_i|alpha, beta tilde.op "Gamma"(alpha, beta), quad i = 1, ..., 10,
$
where $alpha$ and $beta$ are given the hyperpriors
$
  alpha tilde.op "Exp"(1.0) quad beta tilde.op "Gamma"(0.1, 1.0).
$


#figure(
  table(
    columns: (0.3fr, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    // inset: 10pt,
    align: horizon,
    table.header([*Pump*], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]),
    [y], $5$, $1$, $5$, $14$, $3$, $19$, $1$, $1$, $4$, $22$,
    [t], $94.3$, $15.7$, $62.9$, $126.0$, $5.24$, $31.4$, $1.05$, $1.05$, $2.1$, $10.5$,
  ),
  caption: "Number of failures and operating times for ten power plants.",
)<data_power_plants>


In the `stan/pump.stan` file, we write a Stan program for this model. Its defines the data, the parameters and the model, which together specify the likelihood and the priors. Now, we can from R prepare the data and fit the model. In `stan/model.R` we write a script for running four chains for $10000$ iterations, with a burn-in of $2000$. We also utilize the bayesplot package @Gabry_2019 to visualize the results.

// #raw(read("code/stan/pump.stan"), block: true, lang: "stan")
// #raw(read("code/stan/model.R"), block: true, lang: "R")


Fitting the model, @stan-output shows summary statistics for the parameters, including the mean, standard deviation, quantiles, effective sample size and Rhat statistic. All parameters exhibits good convergence, with $hat(R)$ values equal to $1$. Effective sample sizes are large for all parameters (from around 25,000 to 46,000), suggesting low autocorrelation and efficient exploration of the posterior distribution. Monte Carlo standard errors are negligible relative to posterior standard deviations, implying high numerical precision in the estimated posterior summaries.

#figure(
  align(left)[
    #raw(
      "Inference for Stan model: anon_model.
4 chains, each with iter=10000; warmup=2000; thin=1;
post-warmup draws per chain=8000, total post-warmup draws=32000.

           mean se_mean   sd 2.5%  25%  50%   75% 97.5% n_eff Rhat
alpha      0.48    0.00 0.18 0.21 0.35 0.45  0.58  0.89 24685    1
beta       0.29    0.00 0.18 0.05 0.16 0.25  0.38  0.74 25076    1
lambda[1]  0.06    0.00 0.02 0.02 0.04 0.05  0.07  0.12 44146    1
lambda[2]  0.09    0.00 0.08 0.01 0.04 0.07  0.13  0.29 46050    1
lambda[3]  0.09    0.00 0.04 0.03 0.06 0.08  0.11  0.17 43638    1
lambda[4]  0.11    0.00 0.03 0.06 0.09 0.11  0.13  0.18 44393    1
lambda[5]  0.63    0.00 0.34 0.15 0.38 0.57  0.81  1.45 45022    1
lambda[6]  0.62    0.00 0.14 0.37 0.52 0.61  0.70  0.91 43654    1
lambda[7]  1.11    0.00 0.93 0.07 0.44 0.87  1.53  3.49 46891    1
lambda[8]  3.39    0.01 1.65 0.99 2.18 3.12  4.28  7.33 39492    1
lambda[9]  9.37    0.01 2.06 5.80 7.90 9.22 10.68 13.79 35355    1
lambda[10] 0.97    0.00 0.30 0.47 0.76 0.94  1.15  1.65 41896    1",
      block: true,
    )],
  caption: "Output from fitting the Stan model, showing summary statistics for the parameters.",
)<stan-output>



@stan-trace-lambda and @stan-traceplots shows the traceplots for the parameters. The chains seem to have good mixing and no signs of non-convergence. The posterior distributions for the $lambda_i$ are shown in @stan-posterior, and the 95% credible intervals for $lambda_i$ are shown in @stan-intervals. Overall, the results look reasonable given the data, and the inference seems to have worked well. We observe the posteriors of $lambda_(8)$ and $lambda_9$ being relatively high compared to the others, suggesting that these have a higher failure rate.


#figure(
  image("code/stan/trace_lambda.svg", width: 80%),
  caption: [Traceplots for the $lambda_i$ parameters.],
)<stan-trace-lambda>


#figure(
  image("code/stan/traceplots.svg", width: 80%),
  caption: [Traceplots for the $alpha$ and $beta$ hyperparameters.],
)<stan-traceplots>

#figure(
  image("code/stan/posterior_area.svg", width: 80%),
  caption: [Posterior distributions for the $alpha$ and $beta$ hyperparameters.],
)<stan-posterior>

#figure(
  image("code/stan/lambda_intervals.svg", width: 80%),
  caption: [Posterior distributions for the $lambda_i$ parameters, with 95% credible intervals.],
)<stan-intervals>

