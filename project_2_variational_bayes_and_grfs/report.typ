#import "@local/template:1.0.0": *

#let abstract = [
  Project 2 in MA8702
]

#show: template.with(
  title: [Variational Bayes and GRFs],
  author: "Elling Svee",
  date: datetime.today(),
  // bibliography: bibliography("refs.bib", style: "elsevier-harvard"),
  figure-index: (enabled: false),
  table-index: (enabled: false),
  listing-index: (enabled: false),
  table-of-contents: outline(),
  chapter-pagebreak: true,
  bibliography-pagebreak: false,
  fancy-cover-page: false,
  // abstract: abstract,
  language: "en",
)

= Variational Bayes for a simple Gaussian model

Consider the hierarchical model
$
  X_i | mu, gamma & tilde.op cal(N)(mu, 1 \/ gamma), quad i = 1, ..., N \
               mu & tilde.op cal(N)(0, 1 \/ tau), \
            gamma & tilde.op cal(Gamma)(alpha, beta),
$
where $gamma$ and $tau$ denote precision parameters of the Gaussian distribution.

== Joint distribution

For observed data $bold(x) = (x_1, dots, x_N)^top$ and parameters $mu$ and $gamma$, the joint distribution is given by
$
  p(bold(x), mu, gamma) & = p(bold(x)|mu, gamma) p(mu|tau) p(gamma|alpha, beta) \
                        & = ( product_(i=1)^N p(x_i|mu, gamma) ) p(mu|tau) p(gamma|alpha, beta).
$<joint-distribution>

== Exact posterior distribution <exact-posterior>
From Bayes rule, the exact posterior distribution is
$
  p(mu, gamma|bold(x)) & prop p(bold(x), mu, gamma)
$
Using @joint-distribution and inserting the PDFs of the Gaussian and Gamma distributions, we can write the posterior distribution as
$
  p(mu, gamma|bold(x)) & prop
  gamma^(N/2) exp{-gamma / 2 sum_(i=1)^(N) (x_i - mu)^2}
  exp{-tau / 2 mu^2} gamma^(alpha - 1) exp{-beta gamma} \
  & = gamma^(alpha + N/2 - 1) exp{-beta gamma} exp{ -tau/2 mu^2 - gamma / 2 sum_(i=1)^N (x_i - mu)^2}.
$<posterior-before-completing>
Expanding the quadratic term in the exponent, we get
$
  sum_(i=1)^N (x_i - mu)^2 = sum_(i=1)^N x_i^(2) - 2 mu N overline(x) + N mu^2,
$
where $overline(x) = 1/N sum_(i=1)^(N)x_i$. Inserting this into the expression for the posterior distribution, we find
$
  p(mu, gamma|bold(x)) &prop gamma^(alpha + N/2 - 1) exp{-beta gamma} exp{-gamma/2 sum_(i = 1)^(N)x_i^(2)} exp{- 1/2 (gamma N + tau)mu^(2) + gamma N overline(x) mu}.
$
Completing the square in $mu$, we can write the last exponential factor as
$
  exp{- 1/2 (gamma N + tau)(mu - (gamma N overline(x)) / (gamma N + tau))^(2) + (gamma N overline(x))^2 / (2(gamma N + tau))}.
$
For the conditional distribution of $mu$ given $gamma$ and $bold(x)$, the second term in the exponent is constant with respect to $mu$, and we can therefore read off
$
  mu|gamma, bold(x) & tilde.op cal(N)((gamma N overline(x)) / (gamma N + tau), 1 / (gamma N + tau)).
$
Similarly, collecting the terms that depend on $gamma$ from @posterior-before-completing, we find the conditional posterior for $gamma$
$
  p(gamma|mu, bold(x)) prop gamma^(alpha + N/2 - 1) exp{-(beta + 1/2 sum_(i=1)^N (x_i - mu)^2) gamma},
$
which we recognize as
$
  gamma|mu, bold(x) tilde.op cal(Gamma)(alpha + N/2, beta + 1/2 sum_(i=1)^N (x_i - mu)^2).
$
Together, the conditional posteriors show that $p(mu, gamma|bold(x))$ belongs to the Normal--Gamma family, with posterior parameters
$
  mu_n = (gamma N overline(x)) / (gamma N + tau), quad
  tau_n = gamma N + tau, quad
  alpha_n = alpha + N/2, quad
  beta_n = beta + 1/2 sum_(i=1)^N (x_i - mu)^2.
$

== Variational approximation
Assume a mean-field variational approximation
$
  q(mu, gamma) = q (mu) q (gamma).
$
We derive $q$ and $q$ using the variational Bayes update rules. The log of the joint distribution is
$
  log p(bold(x), mu, gamma) prop -gamma/2 sum_(i=1)^N (x_i - mu)^2 - tau/2 mu^2 + (alpha + N/2 - 1) log gamma - beta gamma.
$<log-joint>

From the update rule $log q (mu) prop EE_(q) [log p(bold(x), mu, gamma)]$, we take the expectation over $gamma$ of @log-joint, keeping only terms that depend on $mu$
$
  log q (mu) & prop -EE[gamma]/2 sum_(i=1)^N (x_i - mu)^2 - tau/2 mu^2 \
             & = -EE[gamma]/2 (sum_(i=1)^N x_i^2 - 2 N overline(x) mu + N mu^2) - tau/2 mu^2 \
             & prop -1/2(EE[gamma] N + tau) mu^2 + EE[gamma] N overline(x) mu.
$
This is a quadratic form in $mu$, so $q (mu)$ is Gaussian. Completing the square, we find
$
  q (mu) = cal(N)(nu_q, 1 \/ tau_q),
$
with
$
  tau_q = EE[gamma] N + tau, quad nu_q = (EE[gamma] N overline(x)) / tau_q.
$

Using the update rule $log q (gamma) prop EE_(q) [log p(bold(x), mu, gamma)]$, we take the expectation over $mu$ of @log-joint. Keeping only terms that depend on $gamma$ gives
$
  log q (gamma) prop (alpha + N/2 - 1) log gamma - gamma (beta + 1/2 EE_(q) [sum_(i=1)^N (x_i - mu)^2]).
$
Now, the expected sum of squares is
$
  EE_(q) [sum_(i=1)^N (x_i - mu)^2] = sum_(i=1)^N ((x_i - nu_q)^2 + 1/tau_q) = sum_(i=1)^N (x_i - nu_q)^2 + N / tau_q,
$
where we used $EE[mu] = nu_q$ and $"Var"(mu) = 1\/tau_q$. Therefore $q(gamma)$ has the form of a Gamma distribution
$
  q(gamma) = cal(Gamma)(alpha_q, beta_q),
$
with
$
  alpha_q = alpha + N/2, quad beta_q = beta + 1/2 (sum_(i=1)^N (x_i - nu_q)^2 + N / tau_q).
$

// === CAVI algorithm
//
// The variational parameters are coupled: $tau_q$ and $nu_q$ depend on $EE[gamma] = alpha_q \/ beta_q$, while $beta_q$ depends on $nu_q$ and $tau_q$. We solve this by coordinate ascent variational inference (CAVI), iterating the updates until convergence.
== Evidence lower bound
The ELBO is defined as
$
  cal(L)(q) = underbrace(EE_q [log p(bold(x), mu, gamma)], (1)) - underbrace(EE_q [log q(mu, gamma)], (2)).
$<elbo>
Beginning with $(2)$, we use the mean-field assumption $q(mu, gamma) = q (mu) q (gamma)$. The second term decomposes as
$
  EE_q [log q(mu, gamma)] = EE_(q) [log q (mu)] + EE_(q) [log q (gamma)].
$
For the first term we use $q(mu) = cal(N)(nu_q, 1\/tau_q)$, giving
$
  EE_(q) [log q (mu)] = -1/2 (1 + log(2 pi) - log tau_q).
$
For the Gamma distribution $q(gamma) = cal(Gamma)(alpha_q, beta_q)$ in the second term we get
$
  EE_(q) [log q (gamma)] = alpha_q - log beta_q + log Gamma(alpha_q) + (1 - alpha_q) psi(alpha_q),
$
where $psi$ is the digamma function.



Moving on to $(1)$ in @elbo, we expand the log of the joint distribution into its three components
$
  EE_q [log p(bold(x), mu, gamma)] = EE_q [log p(bold(x)|mu, gamma)] + EE_q [log p(mu|tau)] + EE_q [log p(gamma|alpha, beta)].
$
Using the factorization of $q$ and $EE[gamma] = alpha_q \/ beta_q$, $EE[log gamma] = psi(alpha_q) - log beta_q$, $EE[mu] = nu_q$, $EE[mu^2] = nu_q^2 + 1\/tau_q$, each term evaluates to
$
  EE_q [log p(bold(x)|mu, gamma)] = N/2 (psi(alpha_q) - log beta_q - log(2 pi)) - alpha_q / (2 beta_q) (sum_(i=1)^N (x_i - nu_q)^2 + N / tau_q), \
  EE_q [log p(mu|tau)] = 1/2 (log(tau / (2 pi)) - tau (nu_q^2 + 1 / tau_q)), \
  EE_q [log p(gamma|alpha, beta)] = alpha log beta - log Gamma(alpha) + (alpha - 1)(psi(alpha_q) - log beta_q) - beta alpha_q / beta_q.
$


Combining all terms, we obtain a closed-form expression for the ELBO as a function of the variational parameters.

== Numerical experiment

To assess the quality of the variational approximation, we simulate $N = 100$ observations from a Gaussian distribution with mean $5$ and precision $1$. We choose prior parameters $alpha = 0.01$, $beta = 0.01$ and $tau = 10^(-6)$. @joint-posterior compares the true posterior from @exact-posterior with the variational approximation. Observe clear similarities between the two distributions, although they do not match perfectly. It appears like the variational approximation is slightly more concentrated around the mean than the exact posterior, which is a common feature of mean-field variational approximations.


#figure(
  image("code/figures/vi/joint_posterior.svg", width: 70%),
  caption: [Comparing the exact and variational posteriors],
)<joint-posterior>

== Additional investigation
We experiment with different number of observations $N$ and different prior parameters. In @joint-posterior-other-priors, we use the same number of observations $N = 100$ but different prior parameters $alpha = 1$, $beta = 1$ and $tau = 10^(-4)$. See that there is no notable difference from the previous experiment in @joint-posterior. In @joint-posterior-1000-obs, we increase the number of observations to $N = 1000$ while keeping the original prior parameters. See that the variational approximation is now much closer to the exact posterior.



#subpar.grid(
  figure(image("code/figures/vi/joint_posterior_other_priors.svg"), caption: [
    Different priors
  ]),
  <joint-posterior-other-priors>,

  figure(image("code/figures/vi/joint_posterior_1000_obs.svg"), caption: [
    $1000$ observations
  ]),
  <joint-posterior-1000-obs>,

  columns: (auto, auto),
  caption: [Additional experiments with the variational approximation],
  // label: <grf-simulated-data>,
)

= Gaussian random fields and Kriging

== Setup
For this part of the project, we explore parameter estimation and Kriging for _Gaussian random fields_ (GRFs). Assume a domain $cal(D) = [0, 1]^(2)$ equal to the unit square.

Let $x(dot)$ denote a GRF defined on $cal(D)$, and assume it has the Matérn covariance function
$
  "Cov"(x(bold(s)_i), x(bold(s)_j)) = sigma^(2) (1 - phi h) exp{-phi h},
$
where $h = ||bold(s)_i - bold(s)_j||_2$ is the Euclidean distance between locations $bold(s)_i, bold(s)_j in cal(D)$. Furthermore, assume the mean of the GRF follows the function
$
  mu(bold(s)) = alpha ((s_1 - 0.5) + (s_2 - 0.5)), quad bold(s) = (s_1, s_2)^top, quad alpha in RR.
$
For a set of $N$ locations $bold(s)_1,dots,bold(s)_N in cal(D)$, we observe
$
  y(bold(s)_i) = x(bold(s)_i) + epsilon_i, quad i = 1, dots, N,
$
where $epsilon_i tilde.op cal(N)(0, tau^(2))$ are independent measurement errors.

== Simulation

For the covariance function we select parameters $sigma^2 = 1$, $phi = 10$ and $tau^2 = 0.05^2$, while we let $alpha = 1$ in the mean function. @grf-simulated-data shows a dataset of a simulated GRF on a $100 times 100$ grid, and a corresponding set of observations taken at $N = 200$ random cells in the grid. This is slightly different from the setup in the exercise, as we could also have simulated the GRF directly at the random locations without the need for a grid. However, the grid allows us to visualize the full spatial field, which is useful when we later in @Kriging want to compare the true spatial field with the predictions from Kriging.

#subpar.grid(
  figure(image("code/figures/grfs/grf_full_field.svg"), caption: [
    Full spatial field
  ]),
  figure(image("code/figures/grfs/grf_observations.svg"), caption: [
    $200$ observations
  ]),

  <grf-simulated-observations>,
  columns: (auto, auto),
  caption: [Simulated GRF data],
  label: <grf-simulated-data>,
)

== Parameter estimation

Using the simulated data from @grf-simulated-observations, we estimate the parameters of the GRF model using maximum likelihood estimation (MLE). In the course material, there are derived formulas for how to compute the value and gradient of the log-likelihood analytically. We will therefore not repeat the derivation here. In the exercise, we were asked to optimize the mean $alpha$ and the parameters of the covariance function separately. Analytical formulas for computing the updates are provided in the course material. Instead, I attempt to optimize all parameters jointly using gradient-based optimization. Although this is a different approach, I found it to work well and be more straightforward to implement. The optimization is performed using the Adam optimizer, and gradients are computed using automatic differentiation.

@grf-loglikelihood-history illustrated the convergence of the log-likelihood during optimization. Observe a very rapid convergence, where the optimization stabilizes after only $100$ iterations. The optimized parameters after $200$ iterations were $hat(alpha) = 1.1$, $hat(sigma)^2 = 1.15$, $hat(phi) = 9.49$ and $hat(tau) = 0.06$, which correspond fairly well to the truth.

#figure(
  image("code/figures/grfs/grf_loglikelihood_history.svg", width: 60%),
  caption: [Convergence of log-likelihood during optimization],
)<grf-loglikelihood-history>


== Kriging<Kriging>

Based on the estimated parameters, we perform Kriging to predict the spatial field at a $25 times 25$ grid covering the domain. @kriging-predictions shows the predicted mean and variance. Comparing to the true spatial field in @grf-simulated-data, we see that the mean predictions capture the general structure of the spatial field. The predicted variance is higher in areas with fewer observations. However, as we have a relatively dense set of observations, the predicted variance is generally quite low.


#subpar.grid(
  figure(image("code/figures/grfs/grf_pred_mean.svg"), caption: [
    Predicted mean
  ]),
  figure(image("code/figures/grfs/grf_pred_var.svg"), caption: [
    Predicted variance
  ]),

  columns: (auto, auto),
  caption: [Kriging predictions on $25 times 25$ grid.],
  label: <kriging-predictions>,
)

== Exotic point structures

We can also attempt to repeat the parameter estimation and kriging predictions on a dataset with a more exotic observation structure. @exotic-observation-structure shows the full spatial field and the observations for a dataset where the observations are taken in a cross-shaped pattern in the middle of the domain.  Estimating the parameters, we obtain $hat(alpha) = 1.35$, $hat(sigma)^2 = 0.82$, $hat(phi) = 11.28$ and $hat(tau) = 0.047$. These estimates are notably worse that for the previous dataset, which is expected as our observations are no longer scattered throughout the entire domain.

#subpar.grid(
  figure(image("code/figures/exotic/grf_full_field.svg"), caption: [
    Full spatial field
  ]),
  figure(image("code/figures/exotic/grf_observations.svg"), caption: [
    $200$ observations
  ]),

  columns: (auto, auto),
  caption: [Simulated GRF data],
  label: <exotic-observation-structure>,
)


@exotic-kriging-predictions shows the predictions for the full spatial field based on the cross-shaped observation pattern. See that as we move towards the corners of the domain, the predicted mean becomes more and more inaccurate. Notably, when moving to regions with no observations, the predicted mean approaches the estimated mean $mu(dot)$. For the predicted variance, we see that it increases as we move towards the corners, which is expected as we have fewer observations in those areas.


#subpar.grid(
  figure(image("code/figures/exotic/grf_pred_mean.svg"), caption: [
    Predicted mean
  ]),
  figure(image("code/figures/exotic/grf_pred_var.svg"), caption: [
    Predicted variance
  ]),

  columns: (auto, auto),
  caption: [Kriging predictions on $25 times 25$ grid.],
  label: <exotic-kriging-predictions>,
)
