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

== Robot localization problem
#figure(
  image("figures/robot_in_maze_with_sensors.svg", width: 70%),
)



==
#figure(
  image("figures/robot_with_movement_errors.svg", width: 60%),
)

==
#figure(
  image("figures/robot_with_sensor_error.svg", width: 60%),
)


==
- Robot knows what the map looks like, but not where it is on the map.
- Robot can move, but its movements are also noisy.
- Robot has sensors that can measure the distance to nearby walls, but these measurements are noisy.

*We want to estimate the location of the robot!*


== Let us formalize the problem
For time $t$
- Let $bold(x)_t$ be the state (position and orientation) of the robot
- Let $bold(u)_t$ be the control input (movement commands)
- Let $bold(y)_t$ be the observations (sensor measurements)
Then
$
  bold(x)_t = f(bold(x)_{t-1}, bold(u)_t) + bold(eta)_t
$
and
$
  bold(y)_t = g(bold(x)_t) + bold(epsilon)_t
$
where $bold(eta)_t$ and $bold(epsilon)_t$ have some (possibly non-Gaussian) noise distribution.


== Kalman filter

A Kalman filter is often used, but fails when the relationship between states and observations is non-linear or when the noise is non-Gaussian.

I.e. we would have to assume
$
  bold(x)_t = bold(upright(A)) bold(x)_(t-1) + bold(upright(B)) bold(u)_t + bold(eta)_t, quad bold(eta)_t tilde.op cal(N)(bold(0), bold(upright(Sigma))_bold(eta))
$
and
$
  bold(y)_t = bold(upright(C)) bold(x)_t + bold(epsilon)_t, quad bold(epsilon)_t tilde.op cal(N)(bold(0), bold(upright(Sigma))_bold(epsilon))
$


#focus-slide()[
  We are from importance sampling to particle filters!
]

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


== Just for some sources
@dhayalkar_particle_2025


==
#text(size: 15pt)[
  #bibliography("refs.bib", style: "elsevier-harvard")
]

