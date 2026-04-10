#import "@local/template:1.0.0": *

#show: template.with(
  title: [Sequential Monte Carlo],
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
#let map_fig_width = 80%
#let variances_fig_width = 90%

= Setup
In this project we consider three different filtering approaches for tracking a ship on the surface using two bearings-only observations. There are two sensors measuring angles to a vessel. One sensor is located in east,north coordinate $(0, 0)$, the other at coordinate $(40, 40)$ in km units.

A surface vessel has state $bold(x)_t = (E_t, N_t, v_t, u_t)^top$, where $(E_t, N_t)$ is the east and north position at time $t$ and $(v_t, u_t)$ is the associated velocity vector. The prior initial state is $bold(x)_1 tilde.op cal(N)_4 (bold(mu)_1, bold(upright(Sigma))_1)$, where the expectation is $bold(mu) = (10, 30, 10, -10)^top$ and the covariance matrix is $bold(upright(Sigma)) = "Diag"(10^(2), 10^(2), 5^(2), 5^(2))^top$. The prior dynamic process model for the vessel is defined by
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
for $t = 1, dots, T-1$, $delta = 1 \/ 60$ and $bold(upright(Q))="Diag"(0.1^(2), 0.1^(2), 0.5^(2), 0.5^(2))^top$.

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
Here, the inverse tangent function is defined as indicated in Figure 1 from the project description. Figure ...visualizes the sensor observations as a function of time.

= Extended Kalman Filter

_Implement an extended Kalman filter algorithm. This means linearizing the measurement equation around the predicted state. Plot the filtering solution over time in a map view, along with uncertainty bounds. Also indicate the position of the sensors in the map. Plot the filter variances of the position coordinates over time. Show your derivation and discuss the results._

We begin by implementing an extended Kalman filter algorithm. I follow the steps outlined in the #link("https://en.wikipedia.org/wiki/Extended_Kalman_filter", "Extended Kalman filter") Wikipedia article. The prediction steps are
+ Predicted state estimate: $hat(bold(x))_(t|t-1) = bold(upright(A)) hat(bold(x))_(t-1|t-1)$
+ Predicted covariance estimate: $bold(upright(P))_(t|t-1) = bold(upright(A)) bold(upright(P))_(t-1|t-1) bold(upright(A))^top + bold(upright(Q))$
Define
$
  bold(upright(H))_t = lr((partial h )/ (partial bold(x))|)_(hat(bold(x))_(t|t-1))
$
where $hat(x)_(k|k-1)$ denotes the predicted state estimate. For estimates $hat(E)_t$ and $hat(N)_t$ at some time $t$, we have the matrix
$
  bold(upright(H))_t = mat(
    delim: "[",
    hat(N)_t/(hat(E)_t^2 + hat(N)_t^2), - hat(E)_t/(hat(E)_t^2 + hat(N)_t^2), 0, 0;
  )
$



The update steps are then
+ Measurement residual: $tilde(bold(z))_(t) = bold(y_t) - h(hat(x)_(k|k-1))$
+ Residual covariance: $bold(upright(S))_t = bold(upright(H))_t bold(upright(P))_(t|t-1) bold(upright(H))_t^top + bold(upright(R))$
+ Kalman gain: $bold(upright(P))_(t|t-1) bold(upright(H))_t^top bold(upright(S))_t^(-1)$
+ Updated state estimate: $hat(bold(x))_(t|t) = hat(bold(x))_(t|t-1) + bold(upright(K))_t tilde(bold(z))_t$
+ Updated covariance estimate: $bold(upright(P))_(t|t) = (bold(upright(I)) - bold(upright(K))_(t) bold(upright(H))_t) bold(upright(P))_(t|t-1)$


The filtering algorithm is implemented in `extended_kalman.py`. See that I have used automatic differentiation to obtain the $bold(upright(H))_t$ matrix. Running the algorithm, we obtain the filtering solution and uncertainty bounds bounds in @extended_kalman_solution. We in @extended_kalman_variances plot the variances.

// #figure(
//   image(
//     "code/figures/filter_map.avg",
//     width: 100%
//   ),
//   caption: [
//   Filtering solution for the extended Kalman filter
// ]
// )<extended_Kalman_solution>
//
// #figure(
//   image(
//     "code/figures/filter_variances.avg",
//     width: 100%
//   ),
//   caption: [
//   Filtering variances for the extended Kalman filter
// ]
// )<extended_Kalman_variances>

#subpar.grid(
  figure(image("code/figures/extended_kalman_filter_map.svg", width: map_fig_width), caption: [
    Filtering solution
  ]),
  <extended_kalman_solution>,
  figure(image("code/figures/extended_kalman_filter_variances.svg", width: variances_fig_width), caption: [
    Filtering variances
  ]),
  <extended_kalman_variances>,
  columns: auto,
  caption: [Extended Kalman filter],
  gap: 1.5em,
)





= Particle Filter

Moving on, we implement a standard particle filter algorithm with $B = 10000$ particles. We use the state Markovian process model as proposal at each time step. I follow the steps outlined in Algorithm 5 from #cite(<speekenbrink_tutorial_2016>, form: "prose"), and set $c = 1$ to resample at every time-step. As opposed to the extended Kalman filter, we do not maintain a covariance matrix for the state estimate. The covariance is therefore computed as a weighted sample covariance. The implementation is in `particle.py`. @particle_solution and @particle_variances show the filtering solution and variances for the particle filter, respectively.




#subpar.grid(
  figure(image("code/figures/particle_filter_map.svg", width: map_fig_width), caption: [
    Filtering solution
  ]),
  <particle_solution>,
  figure(image("code/figures/particle_filter_variances.svg", width: variances_fig_width), caption: [
    Filtering variances
  ]),
  <particle_variances>,
  columns: auto,
  caption: [Particle filter],
  gap: 1.5em,
)

= Ensemble Kalman Filter

Lastly, we implement an ensemble Kalman filter with $B = 1000$ particles. I follow the formulas from the basic formulation in the #link("https://en.wikipedia.org/wiki/Ensemble_Kalman_filter", "Ensemble Kalman filter") Wikipedia article.

