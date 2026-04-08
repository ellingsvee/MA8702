#import "@local/presentation:1.0.0": *

#show: presentation.with(
  primary: rgb("#00509e"),
  header-right: none,
  font: "Times New Roman",
  font-size: 22pt,
  language: "en",
  raw-lang: "bash",
)

#title-slide[
  = Particle Filters
  Based on _A tutorial on particle filters_ @speekenbrink_tutorial_2016
]


== What are particle filters?
To understand this, we first need to define the state-space model.

#figure(
  image("figures/state_space_model.png", width: 70%),
)

== State space models
Observable time-series $y_(1:t)$ through a time-series of latent or hidden states $phi_(1:t)$. Two core assumptions:
+ Each observation $y_t$ depends only on the current state $phi_t$.
  $
    p(y_(1:T)|phi_(0:T)) = product_(t=1)^T p(y_t|phi_t).
  $
+ Hidden states change over time according to a first-order Markov process.
$
  p(phi_(0:T)) = p(phi_0) product_(t=1)^T p(phi_t|phi_(t-1)).
$

== Inference for state-space models
We want to estimate the hidden states $phi_(0:t)$ given the observations $y_(1:t)$, i.e. we want to estimate the posterior distribution
$
  p(phi_(0:t)|y_(1:t)) = frac(p(y_(1:t)|phi_(0:t)) p(phi_(0:t)), p(y_(1:t))).
$
See that as $t$ increases, the dimension of the parameter space increases, making inference more costly.


== Inference for state-space models
- We need an algorithm with an approximately *fixed computational cost* at each time point!
#pause
- Particle filter, and more generally *sequential Monte Carlo (SMC)*, is a way to handle this issue.
#pause
- Essentially just *sequential importance sampling* for state-space models, with some resampling to avoid weight degeneracy.

= Application: Robot localization problem
For fun I have built a simple program to simulate a robot moving around in a 2D environment.

#subpar.grid(
  figure(image("figures/autonomous_robot.jpeg")), figure(image("figures/robot_no_particles.png", width: 70%)),
  columns: (1fr, 1fr),
  label: <full>,
)


==

- Robot knows what the map looks like, but not where it is on the map.
- Robot can move, but its movements are also noisy.
- Robot has sensors that can measure the distance to nearby walls, but these measurements are noisy.

#subpar.grid(
  figure(image("figures/robot_with_movement_errors.svg", width: 80%), caption: [
    Movement of sensor can be noisy.
  ]),
  <a>,

  figure(image("figures/robot_with_sensor_error.svg", width: 80%), caption: [
    Sensor measurements can be noisy.
  ]),
  <b>,

  columns: (1fr, 1fr),
  label: <full>,
)

== Let us formalize the problem
Up to time $t$:
- Let $phi_(0:t)$ be the states (position)
- Let $u_(0:t)$ be the control input (movement commands)
- Let $y_(0:t)$ be the observations (sensor measurements)
For functions $f$ and $g$, we have
$
  phi_(t) = f(x_(t-1), u_t) + eta_t
$
and
$
  y_t = g(x_t) + epsilon_t
$
where $eta_t$ and $epsilon_t$ have some iid. and possibly non-Gaussian noise distribution.

#focus-slide()[
  Clearly this is a state-space model!

  Need to estimate the posterior $p(phi_(0:t)|y_(1:t), u_(1:t))$.
]

== What about the Kalman filter?
- A super popular approach to inference in state-space models !
- Assumes that the state transition and observation models are linear, and that the noise is Gaussian.
- We cannot make these assumptions for our problem :-(




= Theory

Instead turn to the particle filter...for this we must start with importance sampling.



== Monte Carlo integration

Want to estimate expectation
$
  EE_p [f(Y)] = integral f(y) p(y) dif y
$

The Monte Carlo estimator is
+ *Sample:* For $1, dots, N$ draw $y^((i)) tilde.op p(y)$
+ *Estimate:*
$
  EE^("MC") = 1/N sum_(i=1)^N f(y^((i)))
$

== Importance sampling
$
  EE_p [f(Y)] = integral p(y)/ q(y) q(y) f(y) dif y = EE_q [ w(Y) f(Y)]
$

Algorithm:
+ *Sample:* For $1, dots, N$ draw $y^((i)) tilde.op q(y)$
+ *Weights:* For $1, dots, N$ compute $w^((i)) = p(y^((i))) \/ q(y^((i)))$
+ *Estimate:*
$
  EE^("IS") = sum_(i=1)^N w^((i)) f(y^((i)))
$

== Self-normalized importance sampling
The $EE^("IS")$ is unbiased, but can have high variance. Instead, we can use the self-normalized estimator
$
  EE^("IS-N") = sum_(i=1)^N W^((i)) f(y^((i)))
$
with normalized weights
$
  W^((i)) = w^((i)) / (sum_(j=1)^N w^((j)))
$

== Sequential importance sampling
Assume we now have a sequence of posterior distributions $p(theta|y_1)$, $p(theta|y_(1:2))$, ..., $p(theta|y_(1:t))$ where $y_(1:t) = (y_1, dots, y_t)$.

To use importance sampling, we need to sample from a proposal distribution $q_t (theta)$  and compute weights
$
  w_t^((i)) = p(theta^((i))|y_(1:t)) / (q_t (theta^((i)))).
$
This becomes computationally expensive as $t$ increases. We need an algorithm with an approximately fixed computational cost at each time point!


== Sequential importance sampling
We can write the weights as
$
  w_t^((i)) = underbrace(frac(p(theta^((i))|y_(1:t)), p(theta^((i))|y_(1:t-1))) dot frac(q_(t-1)(theta^((i))), q_t (theta^((i)))), a_t^((i))) dot underbrace(frac(p(theta^((i))|y_(1:t-1)), q_(t-1)(theta^((i)))), w_(t-1)^((i)))
$

Computing $a_t^((i))$ still requires the $p(theta^((i))|y_(1:t))$ and $q_t (theta^((i)))$. However, there are some cases where we can simplify the computation of this incremental weight.


== Sequential importance sampling

+ The observations are conditionally independent given the parameters
  $
    frac(p(theta^((i))|y_(1:t)), p(theta^((i))|y_(1:t-1))) = p(y_t|theta^((i))) p(y_t|y_(1:t-1))
  $
+ The proposal distribution is the same at each time point
  $q_(t-1)(theta^((i))) \/ q_t (theta^((i))) = 1$

Then the incremental weight is
$
  a_t^((i)) = p(y_t|theta^((i))) p(y_t|y_(1:t-1))
$
Using self-normalized importance sampling, we ignore the $p(y_t|y_(1:t-1))$ term.


== Sequential importance sampling
Algorithm
+ *Initialize:* For $1, dots, N$ draw $theta^((i)) tilde.op q_0(theta)$ and compute normalized weights $W_0^((i)) prop p(theta) \/ q(theta)$ with $sum_(j=1)^(N) W_0^((i)) = 1$.
+ For $t = 1, dots, T$:
  - *Reweigh:* For $1, dots, N$ compute $W_t^((i)) prop p(y_t|theta^((i)))W_(t-1)^((i))$ with $sum_(j=1)^(N) W_t^((i)) = 1$.
  - *Estimate:* Compute the SIS estimate
    $
      EE_t^("SIS-N") = sum_(i=1)^N W_t^((i)) f(theta^((i)))
    $

== Sequential importance sampling
Issue with weight degeneracy: After running an SIS algorithm for a large number of iterations (time points), all but one particle will have negligible weight.

Often measure using the effective sample size
$
  N_"eff" = frac(1, sum_(i=1)^N (W_t^((i)))^2).
$
$N_"eff" = 1$ means all weight is on one particle, while $N_"eff" = N$ means all particles have equal weight.


== Solving the weight degeneracy using resampling

Particles are sampled with replacement from the set of all particles, with a probability that depends on the importance weights.

== Systematic resampling
The most common resampling algorithm is _systematic resampling_. Let ${theta_t, W_t^((i))}$ represent set of particles before resampling, and ${tilde(theta)_t, tilde(W)_t^((i))}$ the set after, we do
+ Draw $u tilde.op "Unif"(0, 1 \/ N)$.
+ Define $U^(i) = (i-1) \/ N + u$ for $i = 1, dots, N$.
+ For $i = 1, dots, N$, find $r$ such that $sum_(k=1)^(r-1) W_t^((k)) <= U^(i) < sum_(k=1)^r W_t^((k))$ and set $j(i) = r$.
+ For $i = 1, dots, N$, set $tilde(theta)_t^((i)) = theta_t^((j(i)))$ and $tilde(W)_t^((i)) = 1 \/ N$.

== Finally...particle filters

We can view the problem of estimating the hidden states $phi_(0:t)$ as estimating a vector of parameters $theta$ which increases in dimension at each time point $t$

By SIS we use proposal distribution $q_t (phi_(0:t)) = q_t (phi_t|phi_(0:t-1)) q_(t-1) (phi_(0:t-1))$. We find the weights
$
  a_t^((i)) = (p(y_t|phi_t^((i))) p(phi_t^((i))|phi_(t-1)^((i)))) / (p(y_t|y_(1:t-1)) q_t (phi_t^((i))|phi_(0:t-1)^((i)))),
$
but the $p(y_t|y_(1:t-1))$ term is ignored when using normalized weights.


== A generic particle filter algorithm



== Further issues and extensions



== Just for some sources
@dhayalkar_particle_2025


==
#text(size: 15pt)[
  #bibliography("refs.bib", style: "elsevier-harvard")
]

