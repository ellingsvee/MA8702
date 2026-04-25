#import "@local/template:1.0.0": *

#show: template.with(
  title: [Ship tracking from bearings],
  author: "Project 3 in MA8702 by Elling Svee",
  date: datetime.today(),
  bibliography: bibliography("refs.bib", style: "elsevier-harvard"),
  figure-index: (enabled: false),
  table-index: (enabled: false),
  listing-index: (enabled: false),
  // table-of-contents: outline(),
  table-of-contents: none,
  chapter-pagebreak: true,
  bibliography-pagebreak: false,
  fancy-cover-page: false,
  // abstract: abstract,
  language: "en",
)

#set figure(gap: 0.1em)
#let map_fig_width = 60%
#let map_fig_width_two = 100%
#let variances_fig_width = 90%

= Setup
In this project we consider three different filtering approaches for tracking a ship on the surface using two bearings-only observations. There are two sensors measuring angles to a vessel. One sensor is located in east,north coordinate $(0, 0)$, the other at coordinate $(40, 40)$ in km units.

A surface vessel has state $bold(x)_t = (E_t, N_t, v_t, u_t)^top$. Here, $(E_t, N_t)$ is the east and north position at time $t$, while $(v_t, u_t)$ is the associated velocity vector. The prior initial state is $bold(x)_1 tilde.op cal(N)_4 (bold(mu)_1, bold(upright(Sigma))_1)$, where the expectation is $bold(mu) = (10, 30, 10, -10)^top$ and the covariance matrix is $bold(upright(Sigma)) = "Diag"(10^(2), 10^(2), 5^(2), 5^(2))^top$. The prior dynamic process model for the vessel is defined by
$
        bold(x)_(t+1) & = bold(upright(A)) bold(x)_t + bold(epsilon)_(t+1), \
     bold(upright(A)) & = mat(
                          delim: "[",
                          1, 0, delta, 0;
                          0, 1, 0, delta;
                          0, 0, 1, 0;
                          0, 0, 0, 1;
                        ), \
  bold(epsilon)_(t+1) & tilde.op cal(N)_4 (bold(0), bold(upright(Q))),
$
with $t = 1, dots, T-1$, $delta = 1 \/ 60$ and $bold(upright(Q))="Diag"(0.1^(2), 0.1^(2), 0.5^(2), 0.5^(2))^top$.

The observations $bold(y)_t = (y_(A, t), y_(B, t))^top$ are made at sensors $A$ and $B$ for times $t = 1, dots, T$. They are modelled as conditionally independent and single-side responsive with additive Gaussian errors
$
  bold(upright(y))_t & = h(bold(x)_t) + bold(epsilon)_t, \
     bold(epsilon)_t & tilde.op cal(N)_2 (bold(0), bold(upright(R))),
$
where $bold(upright(R)) = "Diag"(0.1^(2), 0.1^(2))^top$ and
$
  h(bold(x)_t) =mat(
    delim: "[",
    arctan(E_t \/ N_t);
    arctan((40-N_t) \/ (40-E_t));
  ).
$
The objective for the remainder of the project is to assess the filtering probability density function $p(bold(x)_t|bold(y)_1, dots, bold(y)_t)$ for $t = 1, dots, T$. Datasets for the observations are available at #link("https:// folk.ntnu.no/joeid/MA8702/sensorA.txt", "Sensor A") and #link("https:// folk.ntnu.no/joeid/MA8702/sensorB.txt", "Sensor B").


= Extended Kalman Filter


We begin by implementing an extended Kalman filter algorithm. I follow the steps outlined in #cite(<sarkka_bayesian_2023>, form: "prose") and the #link("https://en.wikipedia.org/wiki/Extended_Kalman_filter", "Extended Kalman filter") Wikipedia article. The main challenge in the extended Kalman filter is to obtain
$
  bold(upright(H))_t = lr((partial h )/ (partial bold(x))|)_(hat(bold(x))_(t|t-1)),
$
which is the linearized $h(dot)$ around the predicted state estimate. I handle this using automatic differentiation through JAX @jax2018github, and therefore avoid the need for manual derivation. The implementation can be found in `extended_kalman.py`.

Running the algorithm, we obtain the filtering solution and uncertainty bounds bounds in @extended_kalman_solution. The thick like is the estimated mean of the filtering distribution, while the ellipses shows an estimated confidence region. The orientation and shape of the ellipses come from the eigenvectors and eigenvalues of the covariance matrix, while the size is scaled to contain a $95%$ probability mass using the chi-squared distribution. Note that the initial state $bold(x)_1 tilde.op cal(N)_4 (bold(mu)_1, bold(upright(Sigma))_1)$ is not included in the plot.

Observe that the ship is estimated to move towards the southeast, while eventually steering almost directly south. The ellipses start out relatively large, but quickly decrease in size. However, observe that the uncertainty in the north-eastern direction becomes larger as the ship moves past the line between the two sensors. This is because the sensors only measure angles, and when directly between the sensors we therefore cannot triangulate the position of the ship.


#figure(image("code/figures/extended_kalman_filter_map.svg", width: map_fig_width), caption: [
  Filtering solution using the extended Kalman filter
])<extended_kalman_solution>




= Particle Filter

Moving on, we implement a standard particle filter algorithm. We use the state Markovian process model as proposal at each time step. I follow the steps outlined in Algorithm 5 from #cite(<speekenbrink_tutorial_2016>, form: "prose"), and set $c = 1$ to resample at each time-step. The implementation is in `particle.py`.

@particle_solution and @particle_solution_B100 show the filtering solution for $B = 1000$ and $B = 100$ particles, respectively. See a clear difference between the performance of the two filters. Whereas the estimated ship trajectory for $B = 1000$ particles is close to the solution obtained from the extended Kalman filter, the solution for $B = 100$ estimates a trajectory that is quite different. A likely cause is that the particle filter with $B = 100$ particles suffers from sample impoverishment. This means that after resampling, many particles have identical states, which reduces the diversity of the particle set and can lead to poor estimates of the filtering distribution.

Another notable observation is that the estimated uncertainty ellipse for the particle filter with $B = 1000$ becomes very small near the end of the trajectory. This is different from the extended Kalman filter. I do not have any good explanation for why this occurs.



#subpar.grid(
  figure(image("code/figures/particle_filter_map.svg", width: map_fig_width_two), caption: [
    $B = 1000$ particles
  ]),
  <particle_solution>,

  figure(image("code/figures/particle_filter_map_B100.svg", width: map_fig_width_two), caption: [
    $B = 100$ particles
  ]),
  <particle_solution_B100>,
  // figure(image("code/figures/particle_filter_variances.svg", width: variances_fig_width), caption: [
  //   Filtering variances
  // ]),
  // <particle_variances>,
  columns: (auto, auto),
  caption: [Filtering solution for the particle filter.],
  gap: 1.5em,
)

= Ensemble Kalman Filter

Lastly, we implement an ensemble Kalman filter. I follow the formulas from the #link("https://en.wikipedia.org/wiki/Ensemble_Kalman_filter", "Ensemble Kalman filter") Wikipedia article and #tc(<understanding_ensemble_kalman_filter>). The implementation is in `ensemble_kalman.py`.

@enkf_solution and @enkf_solution_B100 show the filtering solution for $B = 1000$ and $B = 100$ ensemble members, respectively. Compared to when using different amount of particles in the particle filter, we do not see as much of a difference in the estimated trajectory when using different amount of ensemble members. There appears to be a slight difference in the estimated uncertainty, with the $B=1000$ ellipses being slightly wider. Compared to the extended Kalman filter and particle filters, the estimated uncertainty for the first steps of the trajectory is also a bit bigger for the $B=1000$ case. This likely reflects the differences in how the different filters estimate the covariance.



#subpar.grid(
  figure(image("code/figures/enkf_B1000_stochastic_map.svg", width: map_fig_width_two), caption: [
    $B = 1000$ ensemble members
  ]),
  <enkf_solution>,

  figure(image("code/figures/enkf_B100_stochastic_map.svg", width: map_fig_width_two), caption: [
    $B = 100$ ensemble members
  ]),
  <enkf_solution_B100>,

  columns: (auto, auto),
  caption: [Filtering solution for the ensemble Kalman filter.],
  gap: 1.5em,
)

= Discussion


To summarize, @joint_filter_map shows the filtered trajectories for the different implemented filters. Overall, we see that most of the filters estimate a similar trajectory for the ship, with the exception of the particle filter with $B = 100$ particles.

Overall, I feel like it is difficult to say which of the filters performs best. We know from theory that all the filters can handle the non-linearities in the problem, but the extended Kalman filters might struggle with large non-linearities due to the linearization step. The particle filter is the most general of the three filters, and can handle both non-linearity and non-Gaussianity. It is limited by the number of particles, but for this simple problem the simulation of many particles is not too computationally expensive. The ensemble Kalman filter is more suited to handle high-dimensional problems, but we only have a four-dimensional state space and two-dimensional observations.

For future work, it would be interesting to implement a more advanced particle filter, or to apply the filters for a more complex problem. Although a bit of an overkill for this problem, the Ensemble Transform Kalman Filter mentioned #link("https://www.math.ntnu.no/emner/MA8702/2026v/Jo-seqMC.pdf", "the presentation by Jo") seems like an interesting extension of the ensemble Kalman filter. We could also study the Kalman smoothing distribution, and see how the different filters perform when estimating the smoothing distribution $p(bold(x)_t|bold(y)_1, dots, bold(y)_T)$ for $t = 1, dots, T$.

#figure(image("code/figures/joint_filter_map.svg", width: map_fig_width), caption: [
  Filtering solution for the different implemented filters.
])<joint_filter_map>

