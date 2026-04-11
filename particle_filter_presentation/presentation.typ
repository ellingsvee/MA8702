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

== Possible approaches
/ Extended Kalman filter: Does not work well when the non-linearities are strong, and ssumes Gaussian noise.
/ Ensemble Kalman filter: Can handle non-linearities better, but still assumes Gaussian noise.
/ Particle filter: Can handle non-linearities and non-Gaussian noise, but can be computationally expensive.
/ Others: Any suggestions?


= Theory

== From Gaussian distributions to a cloud of samples
#subpar.grid(
  figure(image("figures/multivariate.png", width: 110%)),
  figure(image("figures/toy.jpg", width: 85%)),
  figure(image("figures/particles_in_maze.png", width: 100%)),

  columns: (auto, auto, auto),
)

== Particles: Probability as a Histogram in Motion
_Each particle is a tiny “grain of probability”, and the collection as a whole shifts and reshapes over time. When new evidence arrives, we adjust the weights of these grains so that the cloud of particles continues to approximate the true probability distribution._ (CITE)

== Particles: More formally

$N$ particles, each with an associated weight
$
  {x_k^((i)), w_k^((i))}_(i=1)^N, quad w_k^((i)) >= 0, quad sum_(i=1)^N w_k^((i)) = 1
$

Approximate the posterior as
$
  p(x_k|z_(1:k)) approx sum_(i=1)^N w_k^((i)) delta(x_k - x_k^((i)))
$

== Particles: More formally
Can estimate expectations as
$
  EE[X_k|z_(1:k)] approx sum_(i=1)^N w_k^((i)) x_k^((i))
$
and the variance as
$
  "Var"[X_k|z_(1:k)] approx sum_(i=1)^N w_k^((i)) (x_k^((i)) - EE[X_k|z_(1:k)])(x_k^((i)) - EE[X_k|z_(1:k)])^T
$

== How to evolve particle filter over time?
Kalman filter:
#figure(
  diagram(
    node-stroke: 1pt,
    node((0, 0), [Predict], corner-radius: 2pt),
    edge("-|>"),
    node((1, 0), [Update], corner-radius: 2pt),
    edge("u,l,d", "--|>"),
  ),
)

Particle filter:
#figure(
  diagram(
    node-stroke: 1pt,
    node((0, 0), [Predict], corner-radius: 2pt),
    edge("-|>"),
    node((1, 0), [Weight], corner-radius: 2pt),
    edge("-|>"),
    node((2, 0), [Resample], corner-radius: 2pt),
    edge("u,ll,d", "--|>"),
  ),
)

== Step 1: Predict - _Carry our beliefs forward_
- At time $k-1$ we have particles ${x_(k-1)^((i)), w_(k-1)^((i))}_(i=1)^N$ approximating $p(x_(k-1)|z_(1:k-1))$.

- Propagate each particle forward by our state transition model
$
  x_k^((i)) = f(x_(k-1)^((i)), u_k) + eta_k^((i))
$
- Note that observation at time $k$ is not used!

== Step 2: Weight - _Incorporate new observation_
- Adjust weights so we approximate "posterior" $p(x_k|z_(1:k))$ instead of "prior" $p(x_k|z_(1:k-1))$.
- Particles consistent with observations get higher weights.
- This will require some work...

== _Importance sampling_
Approximate $p(x)$ using samples from a different distribution $q(x)$
$
  EE[g(x)]_p = integral g(x) p(x) dif x & = integral g(x) p(x)/q(x) q(x) dif x approx 1/N sum_(i=1)^N g(x^((i))) w^((i))
$
where $x^((i)) tilde.op q(x)$ and $w^((i)) = p(x^((i)))/q(x^((i)))$.

// For our context, $p(x)$ represent $p(x_k|z_(1:k))$ and $q(x)$ represent $p(x_k|z_(1:k-1))$.

== _Sequential importance sampling for SSMs_

We can apply importance sampling sequentially at each time step. Something like
$
  w_k^((i)) = p(x_(0:k)|z_(1:k)) / (q_k (x_(0:k)))
$
However, this is not practical because the weights depend on all previous states.

_Be aware that we here now deal with the $x_(0:k)|z_(1:k)$ instead of the marginal $x_(k)|z_(1:k)$!_


== _Sequential importance sampling for SSMs_
#text(size: 20pt)[
  Using the SSM assumptions, we get
  $
    p(x_(0:k)|z_(1:k)) = (p(z_k|x_k) p(x_k|x_(k-1))) / p(z_k|z_(1:k-1)) dot p(x_(0:k-1)|z_(1:k-1))
  $
  and we can choose the proposal distribution so that
  $
    q_k(x_(0:k)) = q_k (x_k|x_(0:k-1)) dot q_(k-1)(x_(0:k-1)).
  $
  Then, the weights can be computed recursively as
  $
    w_k^((i)) = (p(z_k|x_k^((i))) p(x_k^((i))|x_(k-1)^((i)))) / (p(z_k|z_(1:k-1)) q_k (z_k|x_(k-1)^((i)))) dot w_(k-1)^((i))
  $
]


== _Sequential importance sampling for SSMs_
A common proposal distribution is the Bootstrap filter $q_k (x_k|x_(k-1)) = p(x_k|x_(k-1))$. Can also ignore $p(z_k|z_(1:k-1))$ since it is constant across particles. The simplifies weights are
$
  tilde(w)_k^((i)) = p(z_k|x_k^((i))) dot w_(k-1)^((i))
$
which we can normalize to get the final weights
$
  w_k^((i)) = tilde(w)_k^((i)) / (sum_(j=1)^N tilde(w)_k^((j)))
$
The set ${x_k^((i)), w_k^((i))}$ now approximates $p(x_k|z_(1:k))$.


== Step 3: Resampling - _Focusing on what matters_

*Weight degeneracy:* Over time, a few particles may carry most of the total weight.

Effective sample size:
$
  N_"eff" = 1 /(sum_(i=1)^N (w_k^((i)))^2)
$
When
- $N_"eff" = N$: All weights are equal (full diversity)
- $N_"eff" = 1$: One particle has all the weight (complete degeneracy)


== Step 3: Resampling - _Focusing on what matters_

Solution:
- If $N_"eff" \/ N <= c$:
  - Resample according to weights: $P(i^(') = i) = w_k^((i))$.
  - After resampling, set $w_k^((i)) = 1\/N$ for all particles.
The resampled set now is an unweighted approximation of the posterior!

_Sample impoverishment: An issue with resampling is that we reduce the number of unique values present in set of particles._


== Full algorithm

For $k=1$ to $T$:
+ *Predict:* Move particles according to state transition model.
+ *Weight:* Compare predicted measurements with the observation.
+ *Resample:* If necessary, resample/regenerate particles to prevent degeneracy.

Let us finally apply this to our robot localization problem!

// == Briefly: Further issues and extensions
//
// - *Sample impoverishment:* Better resampling strategies to improve estimates of past states.
// - *Inferring time-invariant (static) parameters*
// - Rao-Blackwellized particle filters

= Thank you!

// ==
// #text(size: 15pt)[
//   #bibliography("refs.bib", style: "elsevier-harvard")
// ]

