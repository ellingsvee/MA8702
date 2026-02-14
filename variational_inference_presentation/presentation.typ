#import "@preview/touying:0.6.1": *
#import "@preview/subpar:0.2.2"
#import "@preview/muchpdf:0.1.2": muchpdf
#import "@preview/herodot:0.4.0": *

#import themes.simple: *

#show: simple-theme.with(
  header-right: none,
  primary: rgb("#00509e"),
)

#set text(
  font: "Times New Roman",
  size: 22pt,
)

#set figure(gap: 0.0em)
#let big-text(body) = text(size: 35pt)[#body]

#let fill-color = luma(250)

#show link: set text(fill: blue)


#set raw(lang: "bash")

#set align(horizon)

#show raw.where(block: false): box.with(
  fill: fill-color.darken(5%),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 5pt, x: 2pt),
  radius: 2pt,
)

// Display block code with padding.
#show raw.where(block: true): block.with(
  fill: fill-color.darken(5%),
  inset: (x: 3pt, y: 2pt),
  outset: (x: 0pt, y: 3pt),
  radius: 2pt,
  width: 100%,
)

#let resize-text(body) = layout(size => {
  let font_size = text.size
  let (height,) = measure(
    block(width: size.width, text(size: font_size)[#body]),
  )
  let max_height = size.height

  while height > max_height {
    font_size -= 0.2pt
    height = measure(
      block(width: size.width, text(size: font_size)[#body]),
    ).height
  }

  block(
    height: height,
    width: 100%,
    text(size: font_size)[#body],
  )
})


#title-slide[
  = Variational Inference
  // A Review for Statisticians
  // #cite(<blei_variational_2017>, form: "prose")
]

== Problem
Assume $n$ observations $bold(x)$ and $m$ latent variables $bold(z)$. Bayes rule states that
$
  underbrace(p(bold(z)|bold(x)), "Posterior") = underbrace(p(bold(z)), "Prior") dot frac(overbrace(p(bold(x)|bold(z)), "Likelihood"), underbrace(integral p(bold(x)|bold(z)) p(bold(z)) dif bold(z), "Marginal likelihood")).
$
We want to evaluate the posterior $p(bold(z)|bold(x))$. However, the marginal likelihood $p(bold(x))$ is often hard to compute.

== Variational inference


Variational inference (VI) attempts to _reframe_ the Bayesian inference problem as an optimization problem. Optimization, which involves taking derivatives instead of integrating, is generally faster (and easier?).





= Objective
== Variational objective

// Optimization, which involves taking derivatives instead of integrating, is much easier and generally faster than the latter.

Find a density $q^(ast)(bold(z))$ from a family of densities $cal(D)$ that approximates the posterior
$
  q^(ast)(bold(z)) = op("argmin", limits: #true)_(q(bold(z)) in cal(D)) "KL"(q(bold(z))||p(bold(z)|bold(x))),
$
where the Kullback-Leibler divergence is defined as
$
  "KL"(q(bold(z))||p(bold(z)|bold(x))) = integral q(bold(z)) log frac(q(bold(z)), p(bold(z)|bold(x))) dif bold(z).
$


== Evidence lower bound (ELBO)
Writing out the Kullback-Leibler divergence, we find
$
  "KL"(q(bold(z))||p(bold(z)|bold(x)))
  &= bb(E)_(q(bold(z)))[log q(bold(z))] - bb(E)_(q(bold(z)))[log p(bold(z)|bold(x))] \
  &= bb(E)_(q(bold(z)))[log q(bold(z))] - bb(E)_(q(bold(z)))[log p(bold(z), bold(x))] + bb(E)_(q(bold(z)))[log p(bold(x))] \
  &= bb(E)_(q(bold(z)))[log q(bold(z))] - bb(E)_(q(bold(z)))[log p(bold(z), bold(x))] + underbrace(log p(bold(x)), "Unknown") \
$
#pause
Therefore instead maximize the ELBO, defined as
$
  "ELBO"(q) & = -("KL"(q(bold(z))||p(bold(z)|bold(x)) - log p(bold(x))) \
            & = bb(E)_(q(bold(z)))[log p(bold(z), bold(x))] - bb(E)_(q(bold(z)))[log q(bold(z))].
$

== ELBO intuition

$
  "ELBO"(q) & = bb(E)_(q(bold(z)))[log p(bold(x)|bold(z))] + bb(E)_(q(bold(z)))[log p(bold(z))] - bb(E)_(q(bold(z)))[log q(bold(z))] \
  & = bb(E)_(q(bold(z)))[log p(bold(x)|bold(z))] - bb(E)_(q(bold(z)))[log frac(q(bold(z)), p(bold(z)))] \
  & = underbrace(bb(E)_(q(bold(z)))[log p(bold(x)|bold(z))], "q fits data") - underbrace("KL"(q(bold(z))||p(bold(z))), "q satisfies prior")
$
#pause
Additionally, since
$
  log p(bold(x)) = "ELBO"(q) + "KL"(q(bold(z))||p(bold(z)|bold(x))) > "ELBO"(q),
$
it gives a lower bound for the log marginal likelihood.


= Optimization
==

To make optimization easier, we have to constrain the family of densities $cal(D)$.

// F.ex. we could assume $q(bold(z)) tilde.op cal(N)(bold(mu), bold(upright(Sigma)))$. However, instead of restricting the parametric form of the variational distribution $q(bold(z))$, we usually make an independence assumption.

== Mean-field variational family

Assume latent variables $z_(j)$ are mutually independent, and each governed by their own factor in the variational density
$
  q(bold(z)) = product_(j=1)^m q_j (bold(z)_j).
$
Note, this family cannot model correlations in the posterior distribution. Also see that we do not make any parametric assumptions about the individual $q_(j)(z_(j))$, as this is instead derived for every particular problem.

== Applying mean-field assumption to ELBO
$
  "ELBO"(q) & = bb(E)_(q(bold(z)))[log p(bold(z), bold(x))] - bb(E)_(q(bold(z)))[log q(bold(z))] \
            & = integral product_(i=1)^(m) q_(i)(z_(i)) log p(bold(z), bold(x)) dif bold(z)
              - integral product_(i=1)^(m) q_(i)(z_(i)) log product_(i=1)^(m) q_(i)(z_(i)) dif bold(z).
$

== Optimizing ELBO
#text(size: 20pt)[
  Optimize ELBO with respect to a single variational density $q_(j)(z_(j))$ and assume others are fixed
  $
    "ELBO"(q_(j))
    &= integral product_(i=1)^(m) q_(i)(z_(i)) log p(bold(z), bold(x)) dif bold(z) - integral product_(i=1)^(m) q_(i)(z_(i)) log product_(i=1)^(m) q_(i)(z_(i)) dif bold(z) \
    &prop integral product_(i=1)^(m) q_(i)(z_(i)) log p(bold(z), bold(x)) dif bold(z) - integral q_(j)(z_(j)) log q_(j)(z_(j)) dif z_(j) \
    &= integral q_(j)(z_(j)) (product_(i eq.not j) q_(i)(z_(i)) log p(bold(z), bold(x)) dif bold(z)_(-j)) dif z_(j) - integral q_(j)(z_(j)) log q_(j)(z_(j)) dif z_(j) \
    &= integral q_(j)(z_(j)) bb(E)_(q(bold(z)_(-j)))[log p(bold(z), bold(x))] dif z_(j) - integral q_(j)(z_(j)) log q_(j)(z_(j)) dif z_(j) \
  $
]

== Optimizing ELBO
#text(size: 18pt)[
  To derive the optimal $q_(j)^(ast)(z_(j))$ we follow #cite(<bishop2006pattern>, form: "prose") and define
  $
    log tilde(p) (bold(x), z_(j)) prop bb(E)_(q(bold(z)_(-j)))[log p(bold(z), bold(x))].
  $
  #pause
  Then
  $
    "ELBO"(q_(j))
    &prop integral q_(j)(z_(j)) log tilde(p) (bold(x), z_(j)) dif z_(j) - integral q_(j)(z_(j)) log q_(j)(z_(j)) dif z_(j) \
    &= -integral q_(j)(z_(j)) log frac(q_(j)(z_j), tilde(p)(bold(x), z_j)) dif z_(j)
    = - "KL"(q_(j)(z_(j))||tilde(p)(bold(x), z_(j))) \
  $
  #pause
  Maximizing ELBO with respect to $q_(j)$ is equivalent to minimizing the KL divergence between $q_(j)(z_(j))$ and $tilde(p)(bold(x), z_(j))$! This means that
  $
    q_(j)^(ast)(z_(j)) prop exp (bb(E)_(q(bold(z)_(-j)))[log p(bold(z), bold(x))]).
  $
]


== Coordinate Ascent Variational Inference (CAVI)
This motivates the CAVI algorithm. This is an iterative solution in which we first initialize all factors $q_(j)(z_(j))$ and then cycle through them, updating them conditional on the updates of the other.

Note that
$
  p(z_(j)|bold(z)_(-j), bold(x)) = frac(p(z_j, bold(z)_(-j), bold(x)), p(bold(z)_(-j), bold(x))) prop p(z_j, bold(z)_(-j), bold(x)),
$
meaning we update in terms of the conditional posterior distribution of $z_(j)$ given all other factors $bold(z)_(-j)$.

// =
//
// ==
// // - As we know, MCMC handle this problem through sampling.
// _Variational inference_ (VI) handles this problem through optimization
// $
//   q^(ast)(bold(z)) = "argmin"_(q(bold(z)) in cal(D)) "KL"(q(bold(z)) || p(bold(z)|bold(x))).
// $
// Here, $cal(D)$ is a family of densities over $bold(z)$, and $"KL"(dot)$ denotes the Kullback-Leibler divergence
// $
//   "KL"(q(bold(z)) || p(bold(z)|bold(x))) = bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))] - bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z)|bold(x))].
// $
// However, we still do not know how to compute $p(bold(z)|bold(x))$.
//
// ==
// *Potential benefit of VI*
// + Better utilization of computational resources (e.g., GPU clusters). Therefore scales better to large datasets and complex models.
// + Better convergence properties (e.g., no issues with mixing and autocorrelation). Therefore faster convergence to a good approximation.
//
// *Potential disadvantages of VI*
// + Assumptions about the variational family (e.g., mean-field assumption) may lead to poor approximations.
// + KL divergence might mot be the best measure (f.ex. not symmetric).
//
//
// == MCMC vs. VI
// - MCMC tend to be more computationally intensive that VI, but also provide guarantees of producing (asymptotically) exact samples from target density.
//   - Suitable for small datasets where precise inference is required.
// - VI does not enjoy such guarantees. It can only find a density close to the target. However, tends to be faster than MCMC.
//   - Suitable for large datasets.
//   - Generally underestimates the variance.
//
// == Evidence Lower Bound (ELBO)
// $
//   "KL"(q(bold(z))||p(bold(z)|bold(x)))
//   &= bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))] - bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z)|bold(x))] \
//   &= bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))] - bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z), bold(x))] + bb(E)_(bold(z) tilde.op bold(q))[log p(bold(x))]
// $
// gives us
// $
//   "ELBO"(q) = bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z), bold(x))] - bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))].
// $
//
// == Mean-field variational family
//
// Have to determine the family $cal(D)$ of approximate densities. Assume that the latent variables are mutually independent and each governed by a distinct factor in the variational density.
//
// $
//   q(bold(z)) = product_(j=1)^m q_j (bold(z)_j).
// $
// _This trick also ensures a diagonal covariance matrix..._
//
//
// // = Example: Gaussian mixture model
// //
// // == Example: Gaussian mixture model
// // $K$ clusters, $n$ observations, and $d$-dimensional data.
// // // Latent variables $bold(z) = (bold(mu)^top, bold(c)^top)^(top)$.
// // $
// //                    bold(mu)_(k) & tilde.op cal(N)(bold(0), sigma^(2) bold(upright(I))) quad             &  k = 1, ..., K \
// //                           c_(i) & tilde.op "Categorical"(1 \/ K) quad                                   &  i = 1, ..., n \
// //   bold(x)_(i) | c_(i), bold(mu) & tilde.op cal(N)(c_(i)^(top)bold(mu), sigma^(2) bold(upright(I))) quad & i = 1, ..., n.
// // $
// // The joint density of observed and latent variables is
// // $
// //   p(bold(mu), bold(c), bold(x)) = p(bold(mu)) product_(i=1)^n p(c_(i)) p(bold(x)_(i) | c_(i), bold(mu)).
// // $
//
//
// // == Example: Gaussian mixture model
// //
// // Construct the mean-field variational family
// // $
// //   q(bold(mu), bold(c)) = product_(k=1)^K q_k (bold(mu)_(k); bold(m)_(k), bold(upright(Sigma))_(k)) product_(i=1)^n q_i (c_(i); bold(phi)_(i)),
// // $
// // where
// // $
// //   q_k (bold(mu)_(k); bold(m)_(k), bold(upright(Sigma))_(k)) tilde.op cal(N)(bold(m)_(k), bold(upright(Sigma))_(k))
// //   quad"and"quad
// //   q_i (c_(i); bold(phi)_(i)) tilde.op "Categorical"(bold(phi)_(i)).
// // $
// // What confuses me a bit is the $bold(upright(Sigma))_k$. Do these have to be on the form $bold(upright(Sigma))_k = s_k^(2)bold(upright(I))$?
//
// = Optimization
// == Coordinate Ascent Variational Inference (CAVI)
// Iteratively optimizes each factor of the mean-field variational density, while holding the others fixed.
//
// Splitting $bold(z) = (z_(j), bold(z)_(-j))$ and $q(bold(z)) = q_(j)(z_(j))q_(-j)(bold(z)_(-j))$, and using iterated expectation, we find
// $
//   "ELBO"(q_(j)) prop bb(E)_(j)[bb(E)_(-j)[log p(z_(j), bold(z)_(-j), bold(x))]] - bb(E)_(j)[log q_(j)(z_(j))]
// $
// Maximizing this over densities $q_(j)$ gives the optimal solution (skipping the details)
// $
//   q_(j)^(ast)(z_(j)) prop exp{bb(E)_(-j)[log p(z_(j), bold(z)_(-j), bold(x))]}.
// $

// == Example: Gaussian mixture model
// $
//   "ELBO"(bold(m), bold(s)^(2), bold(phi)) =
//   &sum_(k=1)^(K) bb(E)[log p(bold(mu)_(k)); bold(m)_(k), s_(k)^(2))] \
//   &+ sum_(i=1)^(n) (bb(E)[log p(c_(i) ; phi_(i))] + bb(E)[log p(bold(x)_(i)|c_(i), bold(mu)); bold(phi)_(i), bold(m), bold(s)^(2)]) \
//   &+ sum_(k=1)^(K) bb(E)[log q(bold(mu)_(k); bold(m)_(k), s_(k)^(2))] + sum_(i=1)^(n) bb(E)[log q(c_(i); phi_(i))].
// $
//
//
// == Example: Gaussian mixture model
// #text(size: 18pt)[
//   Update formula for $phi_i$
//   $
//     q_i^(ast)(c_(i); bold(phi)_(i))
//     &prop exp{bb(E)_(bold(c)_(-i), bold(mu))[log p(c_(i), bold(c)_(-i), bold(mu), bold(x))]} \
//     &prop exp{bb(E)_(bold(c)_(-i), bold(mu))[log p(c_(i)) + log p (bold(c)_(-i)) + log p(bold(mu)) + sum_(j=1)^(n) log p(x_(j)|c_(j), bold(mu))]} \
//     &prop exp{bb(E)_(bold(mu))[log p(c_(i)) + log p(x_(i)|c_(i), bold(mu))]} \
//     &prop exp{bb(E)_(bold(mu))[frac(1, K) + log p(bold(x)_(i)|bold(mu)_(c_(i)))]} \
//     &prop exp{bb(E)_(bold(mu))[-frac(1, 2)(bold(x)_(i) - bold(mu)_(c_(i)))^(2)]} \
//     &prop exp{x_(i)m_(c_(i)) - frac(1, 2)s_(c_(i))^(2) - frac(1, 2)m_(c_(i))^(2)} \
//   $
// ]
// == Example: Gaussian mixture model
// #text(size: 18pt)[
//   Update formulas for $m_(k)$ and $s_(k)^(2)$
//   $
//     q_k^(ast)(mu_(k); m_(k), s_(k)^(2))
//     &prop exp{bb(E)_(bold(c), bold(mu)_(-k))[log p(bold(c), mu_(k), bold(mu)_(-k), bold(x))]} \
//     &prop exp{bb(E)_(bold(c), bold(mu)_(-k))[log p (mu_(k)) + sum_(i=1)^(n) log p (x_(i)|c_(i), bold(mu))]} \
//     &prop exp{bb(E)_(bold(c), bold(mu)_(-k))[-frac(1, 2 sigma^(2))mu_(k)^(2) +sum_(i=1)^(n) "I"(c_(i)=k)log p(x_(i)|mu_(k))]} \
//     &prop exp{-frac(1, 2 sigma^(2))mu_(k)^(2) - frac(1, 2)sum_(i=1)^(n) phi_(i, k)(x_(i) - mu_(k))^(2)} \
//     &prop exp{-frac(1, 2)(frac(1, sigma^(2)) + sum_(i=1)^(n) phi_(i, k))[mu_(k) - frac(sum_(i=1)^(n) phi_(i, k)x_(i), frac(1, sigma^(2)) + sum_(i=1)^(n) phi_(i, k))]^(2)} \
//   $
// ]


= Example

== I want to try...linear regression!
#align(center)[
  #image("vi/output/data.svg", width: 55%)
]
==

Assume the following model
- Observations $y tilde.op cal(N)(beta x, sigma^(2))$
- Latent variables $bold(z) = (beta, sigma^(2))$ with priors
  - $beta tilde.op cal(N)(0, tau^(2)sigma^(2))$ ("standardized" so $"Var"[beta \/ sigma] = tau$)
  - $sigma^(2) prop sigma^(-2)$ (improper Jeffreys prior)
// To use CAVI, we must compute variational densities $q^(ast)(beta)$ and $q^(ast)(sigma^(2))$.

== Variational density for $sigma^(2)$:
#text(size: 20pt)[
  Compute
  $
    p (sigma^(2)|bold(y), beta) & prop p(bold(y)|beta, sigma^(2)) p(beta)p(sigma^(2)) \
    & prop (product_(i=1)^(n) (sigma^(2))^(-1/2) exp(-frac(1, 2 sigma^(2))(y_(i) - beta x_(i))^(2))) ((tau^(2)sigma^(2))^(-1/2)exp(-frac(beta, 2 tau^(2)sigma^(2)))) (frac(1, sigma^(2)))\
    &prop (sigma^(2))^(-frac(n+1, 2) - 1) exp(-frac(1, 2 sigma^(2)) underbrace((sum_(i=1)^(n) (y_i - beta x_i)^(2) + frac(beta^(2), tau^(2))), A))
  $
]

== Variational density for $sigma^(2)$:
#text(size: 18pt)[
  We therefore find
  $
    q^(ast)(sigma^(2)) & prop exp(bb(E)_(q(beta))[log p (bold(y), beta, sigma^(2))]) \
                       & prop exp(bb(E)_(q(beta))[log (sigma^(2))^(-frac(n+1, 2)-1) - frac(1, 2 sigma^(2))A]) \
                       & = (sigma^(2))^(-frac(n+1, 2)-1) exp(- frac(1, sigma^(2)) bb(E)_(q(beta))[frac(1, 2)A]) \
  $
  #pause
  Recognize this as the kernel of an inverse-gamma distribution. Letting $nu = bb(E)_(q(beta))[frac(1, 2)A]$ we see
  $
    q^(ast)(sigma^(2)) = frac(nu^(frac(n+1, 2)), Gamma(frac(n+1, 2))) (sigma^(2))^(-frac(n+1, 2)-1) exp(-frac(nu, sigma^(2))).
  $
  Note this depends on $beta$ through the (so far unknown) expectation $bb(E)_(q(beta))[frac(1, 2)A]$.
]

== Variational density for $beta$:

#text(size: 20pt)[
  Compute
  $
    p (beta|bold(y), sigma^(2))
    & prop (product_(i=1)^(n) (sigma^(2))^(-1/2) exp(-frac(1, 2 sigma^(2))(y_(i) - beta x_(i))^(2))) ((tau^(2)sigma^(2))^(-1/2)exp(-frac(beta, 2 tau^(2)sigma^(2)))) (frac(1, sigma^(2)))\
    &prop exp(-frac(1, 2sigma^(2))(beta^(2)(sum_(i=1)^(n) x_i^(2) + frac(1, tau^(2))) - 2beta sum_(i=1)^(n) y_i x_i)) \
    &prop exp(-frac(sum_(i=1)^(n) x_(i)^(2) + frac(1, tau^(2)), 2sigma^(2))(beta - frac(sum_(i=1)^(n) y_(i) x_(i), sum_(i=1)^(n) x_i^(2) + frac(1, tau^(2))))^(2))
  $
]
== Variational density for $beta$:
#text(size: 19pt)[
  Taking expectations
  $
    q^(ast)(beta) &prop exp(bb(E)_(q(sigma^(2)))[log p(beta|bold(y), sigma^(2))]) \
    &prop exp(-frac(sum_(i=1)^(n) x_(i)^(2) + frac(1, tau^(2)), 2)bb(E)_(q(sigma^(2)))[frac(1, sigma^(2))](beta - frac(sum_(i=1)^(n) y_(i) x_(i), sum_(i=1)^(n) x_i^(2) + frac(1, tau^(2))))^(2))
  $
  Inserting the normalizing constants
  $
    q^(ast)(beta) =(2pi underbrace(frac(bb(E)_(q(sigma^(2)))[frac(1, sigma^(2))]^(-1), sum_(i=1)^(n) x_(i)^(2) + frac(1, tau^(2))), sigma_(beta)^(2)))^(-1/2) exp(-frac(sum_(i=1)^(n) x_(i)^(2) + frac(1, tau^(2)), 2)bb(E)_(q(sigma^(2)))[frac(1, sigma^(2))](beta - underbrace(frac(sum_(i=1)^(n) y_(i) x_(i), sum_(i=1)^(n) x_i^(2) + frac(1, tau^(2))), mu_beta))^(2))
  $
]

== Computing the expectations
From $q^(ast)(sigma^(2))$ we needed
$
  nu = bb(E)_(q(beta))[A]
  &= bb(E)_(q(beta))[sum_(i=1)^(n) (y_(i) - beta x_(i))^(2) + frac(beta^(2), tau^(2))] \
  &= sum_(i=1)^(n) y_(i)^(2) - 2 sum_(i=1)^(n) y_(i) x_(i) bb(E)_(q(beta))[beta] + sum_(i=1)^(n) x_i^(2) bb(E)_(q(beta))[beta^(2)] + frac(1, tau^(2)) bb(E)_(q(beta))[beta^(2)]
$
Since $bb(E)_(q(beta))[beta] = mu_beta$ and $bb(E)_(q(beta))[beta^(2)] = mu_beta^(2) + sigma_beta^(2)$, we find
$
  bb(E)_(q(beta))[A] = sum_(i=1)^(n) y_i^(2) - 2 sum_(i=1)^(n) y_i x_i mu_beta + (sigma_(beta)^(2) + mu_(beta)^(2))(sum_(i=1)^(n) x_(i)^(2) + frac(1, tau^(2)))
$

== Computing the expectations
#text(size: 20pt)[
  From $q^(ast)(beta)$ we needed
  $
    bb(E)_(q(sigma^(2)))[frac(1, sigma^(2))]
    &= integral frac(1, sigma^(2)) frac(nu^(frac(n+1, 2)), Gamma(frac(n+1, 2))) (sigma^(2))^(-frac(n+1, 2) - 1) exp(-frac(1, sigma^(2))nu) dif sigma^(2) \
    &= frac(nu^(frac(n+1, 2)), Gamma(frac(n+1, 2))) integral (sigma^(2))^(-(frac(n+1, 2) + 1) - 1) exp(-frac(1, sigma^(2))nu) dif sigma^(2) \
    &= frac(nu^(frac(n+1, 2)), Gamma(frac(n+1, 2))) frac(Gamma(frac(n+1, 2) + 1), nu^(frac(n+1, 2) + 1)) \
    &= frac(n + 1, 2) (frac(1, 2) bb(E)_(q(beta))[A])^(-1)
  $
]
==
Jesus Christ...but we are not done yet! We have to iterate these updates until convergence, and this is measured by the change in ELBO.


== Computing the ELBO
$
  "ELBO"(q) = bb(E)_(q(beta, sigma^(2)))[log p(bold(x)|beta, sigma^(2))] + underbrace(bb(E)_(q(beta, sigma^(2)))[log frac(p(beta, sigma^(2)), q(beta, sigma^(2)))], "still unknown")
$
OK, we skip the details, but this in computable and we can use it to check for convergence.

= Implementation




== Automatic differentiation VI @kucukelbir_automatic_2016
The goal is to work for any model, and only requires that the user specifies $log p (bold(x), bold(z))$.

Implemented in Stan @standev2018rstan!


= Conclusion
== Open problems
- Other distance measures that KL
- Alternatives to mean-field
  - Add dependencies between latent variables (_structured VI_)
  - Mixture of variational densities
- Interface between VI and MCMC
- Statistical properties of VI
- Developing generic VI algorithms that are easy to apply to a wide class of models.



==
#text(size: 15pt)[
  #bibliography("refs.bib", style: "elsevier-harvard")
]

