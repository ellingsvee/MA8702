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



== Just for some sources
@dhayalkar_particle_2025


==
#text(size: 15pt)[
  #bibliography("refs.bib", style: "elsevier-harvard")
]

