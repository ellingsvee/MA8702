#import "@local/template:1.0.0": *

#show: template.with(
  title: [Sequential Monte Carlo],
  author: "Project 3 in MA8702 by Elling Svee",
  date: datetime.today(),
  // bibliography: bibliography("refs.bib", style: "elsevier-harvard"),
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

We begin by implementing an extended Kalman filter algorithm. I follow the steps outlined in the #link("https://en.wikipedia.org/wiki/Extended_Kalman_filter", "Extended Kalman filter") Wikipedia article. Define
$
  bold(upright(H))_t = lr((partial h )/ (partial bold(x))|)_(hat(bold(x))_(t|t-1))
$
where $hat(x)_(k|k-1) = bold(upright(A)) hat(x)_(k-1)$ denotes the predicted state estimate. The prediction steps are
+ Predicted state estimate: $hat(bold(x))_(t|t-1) = bold(upright(A)) hat(bold(x))_(t-1|t-1)$
+ Predicted covariance estimate: $bold(upright(P))_(t|t-1) = bold(upright(A)) bold(upright(P))_(t-1|t-1) bold(upright(A))^top + bold(upright(Q))$
The update steps are
+ Measurement residual: $tilde(bold(z))_(t) = bold(y_t) - h(hat(x)_(k|k-1))$
+ Residual covariance: $bold(upright(S))_t = bold(upright(H))_t bold(upright(P))_(t|t-1) bold(upright(H))_t^top + bold(upright(R))$
+ Kalman gain: $bold(upright(P))_(t|t-1) bold(upright(H))_t^top bold(upright(S))_t^(-1)$
+ Updated state estimate: $hat(bold(x))_(t|t) = hat(bold(x))_(t|t-1) + bold(upright(K))_t tilde(bold(z))_t$
+ Updated covariance estimate: $bold(upright(P))_(t|t) = (bold(upright(I)) - bold(upright(K))_(t) bold(upright(H))_t) bold(upright(P))_(t|t-1)$

= Particle Filter

= Ensemble Kalman Filter

