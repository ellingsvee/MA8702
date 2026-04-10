#import "@local/presentation:1.0.0": *

#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node, shapes


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
  // Based on _A tutorial on particle filters_ @speekenbrink_tutorial_2016
]


== State space models (SSMs)


Observable time-series $z_(1:t)$ through a time-series of latent states $x_(0:t)$.

#let nodes = ("A", "B", "C", "D", "E", "F", "G")
#let edges = (
  (3, 2),
  (4, 1),
  (1, 4),
  (0, 4),
  (3, 0),
  (5, 6),
  (6, 5),
)

#figure(
  diagram(
    // for (i, n) in nodes.enumerate() {
    // 	let θ = 90deg - i*360deg/nodes.len()
    // 	node((θ, 18mm), n, stroke: 0.5pt, name: str(i))
    // }
    // for (from, to) in edges {
    // 	let bend = if (to, from) in edges { 10deg } else { 0deg }
    // 	// refer to nodes by label, e.g., <1>
    // 	edge(label(str(from)), label(str(to)), "-|>", bend: bend)
    // }
    // node-radius: 2em,
    node((0, 0), [$x_0$], stroke: 1pt, radius: 1em),
    edge("-|>"),
    node((1, 0), [$x_1$], stroke: 1pt, radius: 1em, name: "x1"),
    edge("-|>"),
    node((2, 0), [$x_2$], stroke: 1pt, radius: 1em, name: "x2"),
    edge("-|>"),
    node((3, 0), [$dots$], stroke: 0pt, radius: 1em),
    edge("-|>"),
    node((4, 0), [$x_t$], stroke: 1pt, radius: 1em),
    edge((4, 0), (5, 0), "-|>"),

    // Observations
    node((1, 1), [$z_1$], stroke: 1pt, radius: 1em),
    edge((1, 0), (1, 1), "-|>"),
    node((2, 1), [$z_2$], stroke: 1pt, radius: 1em),
    edge((2, 0), (2, 1), "-|>"),
    node((4, 1), [$z_t$], stroke: 1pt, radius: 1em),
    edge((4, 0), (4, 1), "-|>"),
  ),
  // caption: [
  //   State-space model
  // ],
  // gap: 1.5em,
)

Want to estimate $x_t$ given observations $z_(1:t)$.

== State-space models (SSMs)

Core assumptions:
+ Each observation $z_t$ depends only on the current state $x_t$.
  $
    p(z_(1:T)|x_(0:T)) = product_(t=1)^T p(z_t|x_t).
  $
+ Latent states change over time according to a first-order Markov process.
$
  p(x_(0:T)) = p(x_0) product_(t=1)^T p(x_t|x_(t-1)).
$


== Linear and Gaussian example
$
  bold(x)_t & = bold(upright(A)) bold(x)_(t-1) + bold(w)_t, quad & bold(w)_t & tilde.op cal(N)(bold(0), bold(upright(Q))) \
  bold(z)_t & = bold(upright(H))bold(x)_t + bold(v)_t, quad      & bold(v)_t & tilde.op cal(N)(bold(0), bold(upright(R)))
$
In this case, the posterior $p(bold(x)_t|bold(z)_(1:t))$ is Gaussian and can be computed in closed form using the Kalman filter.

*But...how to handle non-linear and/or non-Gaussian SSMs?*





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

== Let us formalize the
Up to time $t$:
- Let $x_(0:t)$ be the states (position)
- Let $u_(0:t)$ be the control input (movement commands)
- Let $z_(1:t)$ be the observations (sensor measurements)
For functions $f$ and $g$, we have
$
  x_(t) = f(x_(t-1), u_t) + eta_t
$
and
$
  z_t = g(x_t) + epsilon_t
$
where $eta_t$ and $epsilon_t$ has some non-Gaussian noise distribution.

=== Approaches for solving the problem
- Extended Kalman filter: Does not work well when the non-linearities are strong, and ssumes Gaussian noise.
- Ensemble Kalman filter: Can handle non-linearities better, but still assumes Gaussian noise.
- Particle filter: Can handle non-linearities and non-Gaussian noise, but can be computationally expensive.


= Theory

== From Gaussian distributions to a cloud of particles
#subpar.grid(
  figure(image("figures/multivariate.png", width: 110%)),
  figure(image("figures/toy.jpg", width: 85%)),
  figure(image("figures/particles_in_maze.png", width: 100%)),

  columns: (auto, auto, auto),
)


==

$N$ particles, each with an associated weight
$
  {x_k^(i), w_k^(i)}_(i=1)^N, quad w_k^(i) >= 0, quad sum_(i=1)^N w_k^(i) = 1
$

Approximate the posterior as
$
  p(x_k|z_(1:k)) approx sum_(i=1)^N w_k^(i) delta(x_k - x_k^(i))
$



== Further issues and extensions
// - *Choice of proposal distribution:* The bootstrap filter can be inefficient when the likelihood is very peaked. Better proposals incorporate the current observation.
// - *Particle impoverishment:* After resampling, many particles are duplicates. Regularization or MCMC moves can help.
// - *Unknown parameters:* Can be handled by augmenting the state space, or using particle MCMC methods.

- *Sample impoverishment and particle smoothing:* TODO
- *Inferring time-invariant (static) parameters:* TODO
- *Rao-Blackwellized particle filters:* TODO

== Discussion
- Which problems in your research could be handled using particle filters?
- How does they relate to other methods you are familiar with, e.g. variational inference, MCMC?
- ...

// ==
// #text(size: 15pt)[
//   #bibliography("refs.bib", style: "elsevier-harvard")
// ]

