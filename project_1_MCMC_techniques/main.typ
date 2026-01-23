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
  title: [MCMC techniques],
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

= Metropolis-Hastings for bivariate densities

In this project we implement three variations of the Metropolis-Hastings (MH). For evaluating the algorithms we consider three bivariate target densities for $bold(x) = (x, y)^"T"$:

/ Gaussian distribution: The first is a bivariate Gaussian distribution with correlation. Its probability density function (PDF) is
  $
    pi(bold(x)) = frac(1, 2 pi det(bold(upright(Sigma)))^(1/2)) exp lr((-frac(1, 2)bold(x)^"T" bold(upright(Sigma))^(-1)bold(x)))
  $
  where $bold(upright(Sigma))$ has $1$ on the diagonal and $0.9$ on the off diagonals.

/ Multimodal density: The second is a multimodal density constructed as a mixture of Gaussian densities. Its PDF is
  $
    pi(bold(x)) = sum_(i=1)^(3) w_i frac(1, 2 pi det(bold(upright(Sigma))_i)^(1/2)) exp lr((-frac(1, 2)(bold(x)-bold(mu)_i)^"T" bold(upright(Sigma))_(i)^(-1)(bold(x) - bold(mu)_i))),
  $
  with $w_i = 1 \/3$ for $i=1, 2, 3$. The means are $bold(mu)_1 = (−1.5, −1.5)^("T")$, $bold(mu)_2 = (1.5, 1.5)^("T")$ and $bold(mu)_3 = (-2, 2)^("T")$, and the covariance matrices all have correlation $0$ and variances $sigma_(1)^(2) = sigma_(2)^(2) = 1$ and $sigma_3^(2) = 0.8$.

/ Volcano density: Lastly we consider a volcano-shaped density with PDF
  $
    pi(bold(x)) prop frac(1, 2 pi) exp lr((-1/2 bold(x)^("T") bold(x)))(bold(x)^("T") bold(x) + 0.25)
  $

@bivariate-densities visualize the three densities on a grid covering $[-5, 5] times [-5, 5]$. The grid spacing is $0.1$, which gives in total $101 times 101$ grid cells.

#figure(
  image("code/output/distributions.svg", width: 100%),
  caption: "Bivariate densities",
)<bivariate-densities>

= Code setup

The three MCMC algorithms are implemented in Python using the JAX @jax2018github library for linear algebra, just-in-time compilation and automatic differentiation (AD). Outside of the main scripts, i have the following utility-files:

/ `densities.py`: Implementation of the three densities.

#raw(read("code/scripts/densities.py"), block: true, lang: "python")

/ `inderence.py`: For running the chains.

#raw(read("code/scripts/inference.py"), block: true, lang: "python")

/ `utils.py`: Computing the autocorrelations.

#raw(read("code/scripts/utils.py"), block: true, lang: "python")

/ `tuning_experiment.py`: Running the algorithms for different $sigma$-values, and plotting results.

#raw(read("code/scripts/tuning_experiment.py"), block: true, lang: "python")



= Random-walk Metropolis-Hastings

The first algorithm we implement is the Random-walk MH.

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

The second algorithm we implement is the Langevin MH.

== Implementation
/ `langevin.py`: Building the Langevin MH kernel
#raw(read("code/scripts/langevin.py"), block: true, lang: "python")

/ `langevin_chain.py`: Running the tuning experiment for the Langevin MH.
#raw(read("code/tasks/langevin_chain.py"), block: true, lang: "python")

== Gaussian distribution

@langevin-guassian shows the tuning experiment for the Gaussian distribution.

#figure(
  image("code/output/langevin/tuning_mvn.svg", width: 100%),
  caption: "Tuning experiment of Langevin MH for Gaussian distribution",
)<langevin-guassian>

== Multimodal distribution

@langevin-multmodal shows the tuning experiment for the multimodal distribution.

#figure(
  image("code/output/langevin/tuning_multimodal.svg", width: 100%),
  caption: "Tuning experiment of Langevin MH for multimodal distribution",
)<langevin-multmodal>

== Volcano distribution

@langevin-volcano shows the tuning experiment for the volcano distribution.

#figure(
  image("code/output/langevin/tuning_volcano.svg", width: 100%),
  caption: "Tuning experiment of Langevin MH for multimodal distribution",
)<langevin-volcano>

= Hamiltonian Metropolis-Hastings

The third algorithm we implement is the Hamiltonian MH.

== Implementation
/ `hamiltonian.py`: Building the Hamiltonian MH kernel
#raw(read("code/scripts/hamiltonian.py"), block: true, lang: "python")

/ `hmc_chain.py`: Running the tuning experiment for the Hamiltonian MH.
#raw(read("code/tasks/hmc_chain.py"), block: true, lang: "python")

== Gaussian distribution

@hmc-guassian shows the tuning experiment for the Gaussian distribution.

#figure(
  image("code/output/hmc/tuning_mvn.svg", width: 100%),
  caption: "Tuning experiment of Hamiltonian MH for Gaussian distribution",
)<hmc-guassian>

== Multimodal distribution

@hmc-multmodal shows the tuning experiment for the multimodal distribution.

#figure(
  image("code/output/hmc/tuning_multimodal.svg", width: 100%),
  caption: "Tuning experiment of Hamiltonian MH for multimodal distribution",
)<hmc-multmodal>

== Volcano distribution

@hmc-volcano shows the tuning experiment for the volcano distribution.

#figure(
  image("code/output/hmc/tuning_volcano.svg", width: 100%),
  caption: "Tuning experiment of Hamiltonian MH for multimodal distribution",
)<hmc-volcano>
