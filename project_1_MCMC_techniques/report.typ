#import "template.typ": *
#import "@preview/cetz:0.4.2"
#import "@preview/muchpdf:0.1.2": muchpdf

// Algorithm
#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

#let py(body) = raw(body, lang: "python")
#let cmd(body) = raw(body, lang: "bash")


#show: ilm.with(
  title: [Metropolis-Hastings for bivariate densities],
  author: "Prosjekt 1 in MA8702. Written by Elling Svee.",
  // date: datetime(year: 2026, month: 01, day: 18),
  date: datetime.today(),
  // abstract: [],
  bibliography: bibliography("refs.bib", style: "elsevier-harvard"),
  figure-index: (enabled: false),
  table-index: (enabled: false),
  listing-index: (enabled: true),
  table-of-contents: none,
  chapter-pagebreak: false,
  fancy-cover-page: true,
)

= Example densities

In this project we implement three variations of the Metropolis-Hastings (MH). For evaluating the algorithms we consider three bivariate target densities for $bold(x) = (x, y)^"T"$:

+ *Gaussian distribution:* The first is a bivariate Gaussian distribution with correlation. Its probability density function (PDF) is
  $
    pi(bold(x)) = frac(1, 2 pi det(bold(upright(Sigma)))^(1/2)) exp lr((-frac(1, 2)bold(x)^"T" bold(upright(Sigma))^(-1)bold(x)))
  $
  where $bold(upright(Sigma))$ has $1$ on the diagonal and $0.9$ on the off diagonals.

+ *Multimodal density:* The second is a multimodal density constructed as a mixture of Gaussian densities. Its PDF is
  $
    pi(bold(x)) = sum_(i=1)^(3) w_i frac(1, 2 pi det(bold(upright(Sigma))_i)^(1/2)) exp lr((-frac(1, 2)(bold(x)-bold(mu)_i)^"T" bold(upright(Sigma))_(i)^(-1)(bold(x) - bold(mu)_i))),
  $
  with $w_i = 1 \/3$ for $i=1, 2, 3$. The means are $bold(mu)_1 = (−1.5, −1.5)^("T")$, $bold(mu)_2 = (1.5, 1.5)^("T")$ and $bold(mu)_3 = (-2, 2)^("T")$, and the covariance matrices all have correlation $0$ and variances $sigma_(1)^(2) = sigma_(2)^(2) = 1$ and $sigma_3^(2) = 0.8$.

+ *Volcano density:* Lastly we consider a volcano-shaped density with PDF
  $
    pi(bold(x)) prop frac(1, 2 pi) exp lr((-1/2 bold(x)^("T") bold(x)))(bold(x)^("T") bold(x) + 0.25)
  $

@bivariate-densities visualize the three densities on a grid covering $[-5, 5] times [-5, 5]$. The grid spacing is $0.1$, which gives in total $101 times 101$ grid cells.

#figure(
  image("code/output/distributions.svg", width: 100%),
  caption: "Bivariate densities",
)<bivariate-densities>

= Theoretical background for Metropolis-Hastings

The MH algorithm is a Markov chain Monte Carlo (MCMC) method for sampling from a target distribution $pi(bold(x))$ known only up to a normalizing constant. Given a current state $bold(x)^((t))$, the algorithm proceeds as follows:

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

The three MCMC algorithms are implemented in Python using the JAX @jax2018github library for linear algebra and automatic differentiation (AD). Outside of the main implementation-scripts, we use the following files:

/ `densities.py`: Implementation of the three densities.

#raw(read("code/scripts/densities.py"), block: true, lang: "python")

/ `inderence.py`: For running the chains.

#raw(read("code/scripts/inference.py"), block: true, lang: "python")

/ `autocorr.py`: Computing the autocorrelations.

#raw(read("code/scripts/autocorr.py"), block: true, lang: "python")

/ `tuning_experiment.py`: Running the algorithms for different $sigma$-values, and plotting results.

#raw(read("code/scripts/tuning_experiment.py"), block: true, lang: "python")


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

== Implementation
/ `random_walk.py`: Building the Random-walk MH kernel
#raw(read("code/scripts/random_walk.py"), block: true, lang: "python")

/ `rwmh_chain.py`: Running the tuning experiment for the Random-walk MH.
#raw(read("code/tasks/rwmh_chain.py"), block: true, lang: "python")

== Gaussian distribution

@rwmh-guassian shows the tuning experiment for the Gaussian distribution. Observe that $sigma = 1.5$ gives an acceptance rate of $0.264$. This is closest to the theoretical optimal, and is therefore preferred.

#figure(
  image("code/output/rwmh/tuning_mvn.svg", width: 100%),
  caption: "Tuning experiment of Random-walk MH for Gaussian distribution",
)<rwmh-guassian>

== Multimodal distribution

@rwmh-multimodal shows the tuning experiment for the multimodal distribution. $sigma = 1.5$ gives an acceptance rate of $0.503$. This is closest to the theoretical optimal, and is therefore preferred.

#figure(
  image("code/output/rwmh/tuning_multimodal.svg", width: 100%),
  caption: "Tuning experiment of Random-walk MH for multimodal distribution",
)<rwmh-multimodal>

== Volcano distribution

@rwmh-volcano shows the tuning experiment for the volcano distribution. $sigma = 1.5$ gives an acceptance rate of $0.523$. This is closest to the theoretical optimal, and is therefore preferred.

#figure(
  image("code/output/rwmh/tuning_volcano.svg", width: 100%),
  caption: "Tuning experiment of Random-walk MH for volcano distribution",
)<rwmh-volcano>



= Langevin Metropolis-Hastings

== Theory

The Metropolis-adjusted Langevin algorithm (MALA) incorporates gradient information into the proposal. It is motivated by the _Langevin diffusion_, the stochastic differential equation
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

== Implementation
/ `langevin.py`: Building the Angevin MH kernel
#raw(read("code/scripts/langevin.py"), block: true, lang: "python")

/ `langevin_chain.py`: Running the tuning experiment for the Angevin MH.
#raw(read("code/tasks/langevin_chain.py"), block: true, lang: "python")

== Gaussian distribution

@langevin-guassian shows the tuning experiment for the Gaussian distribution. $sigma=0.5$ is closest to the optimal acceptance probability, and is therefore preferred.

#figure(
  image("code/output/langevin/tuning_mvn.svg", width: 100%),
  caption: "Tuning experiment of Langevin MH for Gaussian distribution",
)<langevin-guassian>

== Multimodal distribution

@langevin-multmodal shows the tuning experiment for the multimodal distribution. $sigma=1.0$ is closest to the optimal acceptance probability, and is therefore preferred.

#figure(
  image("code/output/langevin/tuning_multimodal.svg", width: 100%),
  caption: "Tuning experiment of Langevin MH for multimodal distribution",
)<langevin-multmodal>

== Volcano distribution

@langevin-volcano shows the tuning experiment for the volcano distribution. $sigma=1.5$ is closest to the optimal acceptance probability, and is therefore preferred.

#figure(
  image("code/output/langevin/tuning_volcano.svg", width: 100%),
  caption: "Tuning experiment of Langevin MH for multimodal distribution",
)<langevin-volcano>

= Hamiltonian Metropolis-Hastings

== Theory

Hamiltonian Monte Carlo (HMC) augments the target with auxiliary _momentum_ variables $bold(p) in RR^d$ and samples from the joint distribution
$
  pi(bold(x), bold(p)) prop pi(bold(x)) exp lr((-frac(1, 2) bold(p)^"T" bold(p))).
$
This defines a Hamiltonian system with potential energy $U(bold(x)) = -log pi(bold(x))$ and kinetic energy $K(bold(p)) = frac(1, 2) bold(p)^"T" bold(p)$, giving total energy (Hamiltonian)
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

== Implementation
/ `hamiltonian.py`: Building the HMH kernel
#raw(read("code/scripts/hamiltonian.py"), block: true, lang: "python")

/ `hmc_chain.py`: Running the tuning experiment for the HMC.
#raw(read("code/tasks/hmc_chain.py"), block: true, lang: "python")

== Gaussian distribution

@hmc-guassian shows the tuning experiment for the Gaussian distribution. For Hamiltonian MC we prefer a high acceptance rate. Looking at the plots we see that $sigma=0.5$ gives the smallest correlation between consecutive samples. As it still has a very high acceptance rate, this is preferred.

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
  caption: "Tuning experiment of HMC for multimodal distribution",
)<hmc-volcano>
