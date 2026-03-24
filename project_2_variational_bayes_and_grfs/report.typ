#import "@local/template:1.0.0": *

#show: template.with(
  title: [Variational Bayes and GRFs],
  author: "Project 2 in MA8702 by Elling Svee",
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
$
We can insert the PDFs for the three distributions, which gives
$
  p(bold(x), mu, gamma)
  &= ( product_(i=1)^N (gamma / (2 pi))^(1/2) exp{-gamma / 2 (x_i - mu)^2} ) (tau / (2 pi))^(1/2) exp{-tau / 2 mu^2} gamma^(alpha - 1) exp{-beta gamma} \
  &prop gamma^(N/2) exp{-gamma / 2 sum_(i=1)^N (x_i - mu)^2} exp{-tau / 2 mu^2} gamma^(alpha - 1) exp{-beta gamma} \
  & = gamma^(alpha + N/2 - 1) exp{-beta gamma} exp{ -tau/2 mu^2 - gamma / 2 sum_(i=1)^N (x_i - mu)^2}.
$<joint-distribution>

== Exact posterior distribution <exact-posterior>
From Bayes rule, the exact posterior distribution is
$
  p(mu, gamma|bold(x)) & prop p(bold(x), mu, gamma)
$
We in @joint-distribution already found $p(bold(x), mu, gamma)$. Expanding the quadratic term in the exponent, we get
$
  sum_(i=1)^N (x_i - mu)^2 = sum_(i=1)^N x_i^(2) - 2 mu N overline(bold(x)) + N mu^2,
$
where $overline(bold(x)) = 1/N sum_(i=1)^(N)x_i$. Inserting this into the expression for the posterior distribution gives
$
  p(mu, gamma|bold(x)) &prop gamma^(alpha + N/2 - 1) exp{-beta gamma} exp{-gamma/2 sum_(i = 1)^(N)x_i^(2)} exp{- 1/2 (gamma N + tau)mu^(2) + gamma N overline(bold(x)) mu}.
$<posterior-expansion>
Now, do derive expressions for the posterior parameters, we can collect the terms that depend on $mu$ and $gamma$ respectively.

Completing the square in $mu$, we can write the last exponential factor in @posterior-expansion as
$
  exp{- 1/2 (gamma N + tau)(mu - (gamma N overline(bold(x))) / (gamma N + tau))^(2) + (gamma N overline(bold(x)))^2 / (2(gamma N + tau))}.
$
For the conditional distribution of $mu|gamma, bold(x)$, the second term in the exponent is constant with respect to $mu$, and we can therefore read off
$
  mu|gamma, bold(x) & tilde.op cal(N)((gamma N overline(bold(x))) / (gamma N + tau), 1 / (gamma N + tau)).
$
Similarly, collecting the terms that depend on $gamma$, we from @joint-distribution find the posterior
$
  p(gamma|mu, bold(x)) prop gamma^(alpha + N/2 - 1) exp{-(beta + 1/2 sum_(i=1)^N (x_i - mu)^2) gamma},
$
which we recognize as
$
  gamma|mu, bold(x) tilde.op cal(Gamma)(alpha + N/2, beta + 1/2 sum_(i=1)^N (x_i - mu)^2).
$
Together, the conditional posteriors show that $p(mu, gamma|bold(x))$ belongs to the Normal-Gamma family. Here $mu|gamma, bold(x) tilde.op cal(N)(mu_n, 1 \/ tau_n)$ and $gamma|mu, bold(x) tilde.op cal(Gamma)(alpha_n, beta_n)$ with parameters
$
  mu_n = (gamma N overline(bold(x))) / (gamma N + tau), quad
  tau_n = gamma N + tau, quad
  alpha_n = alpha + N/2, quad
  beta_n = beta + 1/2 sum_(i=1)^N (x_i - mu)^2.
$

== Variational approximation
Assume a mean-field variational approximation
$
  q(mu, gamma) = q (mu) q (gamma).
$
We derive $q(mu)$ and $q(gamma)$ using the variational Bayes update rules
$
  log q (mu) prop EE_(gamma) [log p(bold(x), mu, gamma)],
$<VB-update-mu>
and
$
  log q (gamma) prop EE_(mu) [log p(bold(x), mu, gamma)].
$<VB-update-gamma>

The log of the joint distribution is
$
  log p(bold(x), mu, gamma) prop -gamma/2 sum_(i=1)^N (x_i - mu)^2 - tau/2 mu^2 + (alpha + N/2 - 1) log gamma - beta gamma.
$<log-joint>

From @VB-update-mu, we take the expectation over $gamma$ of @log-joint, keeping only terms that depend on $mu$
$
  log q (mu) & prop -EE[gamma]/2 sum_(i=1)^N (x_i - mu)^2 - tau/2 mu^2 \
             & = -EE[gamma]/2 (sum_(i=1)^N x_i^2 - 2 N overline(bold(x)) mu + N mu^2) - tau/2 mu^2 \
             & prop -1/2(EE[gamma] N + tau) mu^2 + EE[gamma] N overline(bold(x)) mu.
$
This is a quadratic form in $mu$, so $q (mu)$ is Gaussian. Completing the square, we find
$
  q (mu) = cal(N)(nu_q, 1 \/ tau_q),
$
with
$
  tau_q = EE[gamma] N + tau, quad nu_q = (EE[gamma] N overline(bold(x))) / tau_q.
$

For $q(gamma)$ the update rule from @VB-update-gamma. Taking the expectation over $mu$ of @log-joint and keeping only terms that depend on $gamma$ gives
$
  log q (gamma) prop (alpha + N/2 - 1) log gamma - gamma (beta + 1/2 EE_(mu) [sum_(i=1)^N (x_i - mu)^2]).
$
Now, the expected sum of squares is
$
  EE_(mu) [sum_(i=1)^N (x_i - mu)^2] = sum_(i=1)^N ((x_i - nu_q)^2 + 1/tau_q) = sum_(i=1)^N (x_i - nu_q)^2 + N / tau_q,
$
where we used $EE[mu] = nu_q$ and $"Var"(mu) = 1\/tau_q$. Therefore $q(gamma) tilde.op Gamma(alpha_q, beta_q)$ with
$
  alpha_q = alpha + N/2, quad beta_q = beta + 1/2 (sum_(i=1)^N (x_i - nu_q)^2 + N / tau_q).
$

== Evidence lower bound
The ELBO is defined as
$
  cal(L)(q) = underbrace(EE_q [log p(bold(x), mu, gamma)], (1)) - underbrace(EE_q [log q(mu, gamma)], (2)).
$<elbo>
Beginning with $(2)$, we use the mean-field assumption $q(mu, gamma) = q (mu) q (gamma)$ to decompose
$
  EE_q [log q(mu, gamma)] = EE_(q) [log q (mu)] + EE_(q) [log q (gamma)].
$
As $q(mu) = cal(N)(nu_q, 1\/tau_q)$ we have
$
  EE_(q) [log q (mu)] = -1/2 (1 + log(2 pi) - log tau_q),
$
and $q(gamma) = cal(Gamma)(alpha_q, beta_q)$ gives
$
  EE_(q) [log q (gamma)] = alpha_q - log beta_q + log Gamma(alpha_q) + (1 - alpha_q) psi(alpha_q)
$
where $psi$ is the digamma function.



Moving on to $(1)$ from @elbo, we expand the log of the joint distribution into its three components
$
  EE_q [log p(bold(x), mu, gamma)] = EE_q [log p(bold(x)|mu, gamma)] + EE_q [log p(mu|tau)] + EE_q [log p(gamma|alpha, beta)].
$
Using
$
  EE[gamma] = alpha_q \/ beta_q, quad EE[log gamma] = psi(alpha_q) - log beta_q
$
and
$
  EE[mu] = nu_q, quad & EE[mu^2] = nu_q^2 + 1\/tau_q,
$
we evaluate
$
  EE_q [log p(bold(x)|mu, gamma)] = N/2 (psi(alpha_q) - log beta_q - log(2 pi)) - alpha_q / (2 beta_q) (sum_(i=1)^N (x_i - nu_q)^2 + N / tau_q), \
  EE_q [log p(mu|tau)] = 1/2 (log(tau / (2 pi)) - tau (nu_q^2 + 1 / tau_q)), \
  EE_q [log p(gamma|alpha, beta)] = alpha log beta - log Gamma(alpha) + (alpha - 1)(psi(alpha_q) - log beta_q) - beta alpha_q / beta_q.
$


Combining all terms, we obtain a closed-form expression for the ELBO as a function of the variational parameters.

== Numerical experiment

To assess the quality of the variational approximation, we simulate $N = 100$ observations from a Gaussian distribution with mean $5$ and precision $1$. We choose prior parameters $alpha = 0.01$, $beta = 0.01$ and $tau = 10^(-6)$.

@joint-posterior plots the approximate posterior $q(mu, gamma)$ and the true posterior from @exact-posterior. Observe clear similarities between the two, although they do not match perfectly. It appears like the variational approximation is slightly more concentrated around the mean than the exact posterior, which is a common feature of mean-field variational approximations.


#figure(
  image("code/figures/vi/joint_posterior.svg", width: 70%),
  caption: [Comparing the exact and variational posteriors],
)<joint-posterior>

== Additional investigation
We experiment with different number of observations $N$ and different prior parameters. In @joint-posterior-other-priors, we use the same number of observations $N = 100$, but different prior parameters $alpha = 1$, $beta = 1$ and $tau = 10^(-4)$. See that there is no notable difference from the previous experiment in @joint-posterior. In @joint-posterior-1000-obs, we increase the number of observations to $N = 1000$ while keeping the original prior parameters. See that the variational approximation is now much closer to the exact posterior.



#subpar.grid(
  figure(image("code/figures/vi/joint_posterior_other_priors.svg"), caption: [
    Priors $(alpha, beta, tau) = (1, 1, 10^(-4))$
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

== Implementation

The full implementation for this task is found in #cmd("code/run_vi.py"),

= Gaussian random fields and Kriging


== Setup
For this part of the project, we explore parameter estimation and Kriging for _Gaussian random fields_ (GRFs). Assume a domain $cal(D) = [0, 1]^(2)$. Let $x(dot)$ denote a GRF defined on $cal(D)$, and assume it has the Matérn covariance function
$
  "Cov"(x(bold(s)_i), x(bold(s)_j)) = sigma^(2) (1 - phi h) exp{-phi h}, quad bold(s)_i, bold(s)_j in cal(D),
$<covariance-function>
where $h = ||bold(s)_i - bold(s)_j||_2$ is the Euclidean distance. Furthermore, assume the mean of the GRF follows the function
$
  mu(bold(s)) = alpha ((s_1 - 0.5) + (s_2 - 0.5)), quad bold(s) = (s_1, s_2)^top in cal(D), quad alpha in RR.
$<mean-function>
For a set of $N$ locations $bold(s)_1,dots,bold(s)_N in cal(D)$, we observe
$
  y(bold(s)_i) = x(bold(s)_i) + epsilon_i, quad i = 1, dots, N,
$
where $epsilon_i tilde.op cal(N)(0, tau^(2))$ are independent measurement errors.

== Simulation

For the covariance function we select parameters $sigma^2 = 1$, $phi = 10$ and $tau^2 = 0.05^2$, while we let $alpha = 1$ in the mean function. @grf-simulated-data shows a dataset of a simulated GRF on a $100 times 100$ grid, and a corresponding set of observations taken at $N = 200$ random cells in the grid. Note that this is slightly different from the setup described in the project, as we could also have simulated the GRF directly at the random locations without needing a grid. However, the grid allows us to visualize the full spatial field, which is useful when we later in @Kriging want to compare the true spatial field with the Kriging predictions.

#subpar.grid(
  figure(image("code/figures/grfs/grf_full_field.svg"), caption: [
    Full spatial field
  ]),
  figure(image("code/figures/grfs/grf_observations.svg"), caption: [
    Observations
  ]),

  <grf-simulated-observations>,
  columns: (auto, auto),
  caption: [Simulated GRF data on a $100 times 100$ grid, and $200$ observations taken at random locations in the grid.],
  label: <grf-simulated-data>,
)

== Parameter estimation

Using the simulated data from @grf-simulated-observations, we estimate the parameters of the GRF model using maximum likelihood estimation (MLE). In the #link("https://www.math.ntnu.no/emner/MA8702/2026v/gaussianProcesses.pdf", "course material"), there are derived formulas for how to compute the value and gradient of the log-likelihood analytically. We will therefore not repeat this derivation here. Other than differentiating the mean- and covariance functions from @mean-function and @covariance-function with respect to our parameters, the only thing we need to do is to implement the update rules.


As an alternative to the analytical approach, I instead attempt to optimize the parameters jointly using gradient-based optimization. I am aware that this is not the approach described in the project, but hopefully it is still acceptable as it is a valid approach to parameter estimation. Implementing the joint optimization is straightforward using automatic differentiation. We can write a function that computes the log-likelihood of the observed data given the parameters, and then use an optimizer such as Adam to optimize the parameters.

@grf-loglikelihood-history illustrated the convergence of the log-likelihood during optimization. Observe a very rapid convergence, where the optimization stabilizes after approximately $50$ iterations. The optimized parameters after $200$ iterations were $hat(alpha) = 1.1$, $hat(sigma)^2 = 1.15$, $hat(phi) = 9.5$ and $hat(tau) = 0.06$, which correspond fairly well to the truth.

#figure(
  image("code/figures/grfs/grf_loglikelihood_history.svg", width: 60%),
  caption: [Convergence of log-likelihood during optimization],
)<grf-loglikelihood-history>


== Kriging<Kriging>

Based on the estimated parameters, we perform Kriging to predict the spatial field at a $25 times 25$ grid covering the domain. Again, we rely on the formulas from the course material to compute the predicted mean and variance at each location in the grid.

@kriging-predictions shows the predicted mean and variance. Comparing to the true spatial field in @grf-simulated-data, we see that the mean predictions capture the general structure of the spatial field. The predicted variance is higher in areas with fewer observations, and close to 0 near observed locations. Note that as we have a relatively dense set of observations, the predicted variance is generally quite low.


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

We can also attempt to repeat the parameter estimation and kriging predictions on a dataset with a more exotic observation structure. @exotic-observation-structure shows the full spatial field and the observations for a dataset where $200$ observations are taken in a cross-shaped pattern in the middle of the domain.

Estimating the parameters, we obtain $hat(alpha) = 1.35$, $hat(sigma)^2 = 0.82$, $hat(phi) = 11.28$ and $hat(tau) = 0.047$. Interestingly, we see that although we have the same number of observations as in the previous dataset, the parameter estimates are worse. This is likely because the observations are more clustered together, which makes it more difficult to capture the spatial structure of the field. However, the estimates are still reasonably close to the truth, meaning that the optimization was able to find a good solution despite the more challenging observation structure.

#subpar.grid(
  figure(image("code/figures/exotic/grf_full_field.svg"), caption: [
    Full spatial field
  ]),
  figure(image("code/figures/exotic/grf_observations.svg"), caption: [
    Observations
  ]),

  columns: (auto, auto),
  caption: [Simulated GRF data on a $100 times 100$ grid, and $200$ observations taken in a cross shaped pattern.],
  label: <exotic-observation-structure>,
)


@exotic-kriging-predictions shows the predictions for the full spatial field based on the cross-shaped observation pattern. For the predicted mean, we from @exotic-kriging-mean see that the predictions are quite accurate in the observed parts of the domain, but become increasingly inaccurate as we move towards the corners. Notably, the predicted mean approaches the estimated mean in the corners, resulting a in much "smoother" prediction compared to the true spatial field. As from the previous example, we for the predicted variance see from @exotic-kriging-var that the predicted variance is close to 0 in the observed areas, but increases as we move towards the corners. However, the predicted variance is generally higher than previously, which aligns with the intuition that we are more uncertain in regions far from any observations.

#subpar.grid(
  figure(image("code/figures/exotic/grf_pred_mean.svg"), caption: [
    Predicted mean
  ]),
  <exotic-kriging-mean>,

  figure(image("code/figures/exotic/grf_pred_var.svg"), caption: [
    Predicted variance
  ]),
  <exotic-kriging-var>,

  columns: (auto, auto),
  caption: [Kriging predictions on $25 times 25$ grid.],
  label: <exotic-kriging-predictions>,
)

== Implementation

The implementations for this task are found in
- #cmd("code/spatial.py"): Implementation of the GRF simulation, parameter estimation and Kriging prediction.
- #cmd("code/run_grf.py"): Script to run for the setup with random-observations.
- #cmd("code/run_exotic.py"): Script to run for the setup with cross-shaped observation.
