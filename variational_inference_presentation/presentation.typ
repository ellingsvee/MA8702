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
  = Variational Inference: A Review for Statisticians
  #cite(<blei_variational_2017>, form: "prose")
]

== Problem

Joint density of $m$ latent variables $bold(z)$ and n observations $bold(x)$
$
  p(bold(z), bold(x)) = p(bold(z)) p(bold(x)|bold(z)).
$
Want to approximate the posterior
$
  p(bold(z)|bold(x)) = frac(p(bold(z), bold(x)), p(bold(x))) quad "where" p(bold(x)) = integral p(bold(z), bold(x)) d bold(z).
$
However, $p(bold(x))$ is often hard to compute...



=

==
// - As we know, MCMC handle this problem through sampling.
_Variational inference_ (VI) handles this problem through optimization
$
  q^(ast)(bold(z)) = "argmin"_(q(bold(z)) in cal(D)) "KL"(q(bold(z)) || p(bold(z)|bold(x))).
$
Here, $cal(D)$ is a family of densities over $bold(z)$, and $"KL"(dot)$ denotes the Kullback-Leibler divergence
$
  "KL"(q(bold(z)) || p(bold(z)|bold(x))) = bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))] - bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z)|bold(x))].
$
However, we still do not know how to compute $p(bold(z)|bold(x))$.

==
*Potential benefit of VI*
+ Better utilization of computational resources (e.g., GPU clusters). Therefore scales better to large datasets and complex models.
+ Better convergence properties (e.g., no issues with mixing and autocorrelation). Therefore faster convergence to a good approximation.

*Potential disadvantages of VI*
+ Assumptions about the variational family (e.g., mean-field assumption) may lead to poor approximations.
+ KL divergence might mot be the best measure (f.ex. not symmetric).


== Evidence Lower Bound (ELBO)
$
  "KL"(q(bold(z))||p(bold(z)|bold(x)))
  &= bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))] - bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z)|bold(x))] \
  &= bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))] - bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z), bold(x))] + bb(E)_(bold(z) tilde.op bold(q))[log p(bold(x))]
$
gives us
$
  "ELBO"(q) = bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z), bold(x))] - bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))].
$

== Mean-field variational family

Have to determine the family $cal(D)$ of approximate densities. Assume that the latent variables are mutually independent and each governed by a distinct factor in the variational density.

$
  q(bold(z)) = product_(j=1)^m q_j (bold(z)_j).
$
_This trick also ensures a diagonal covariance matrix..._


= Example: Gaussian mixture model

== Example: Gaussian mixture model
$K$ clusters, $n$ observations, and $d$-dimensional data.
// Latent variables $bold(z) = (bold(mu)^top, bold(c)^top)^(top)$.
$
                   bold(mu)_(k) & tilde.op cal(N)(bold(0), sigma^(2) bold(upright(I))) quad             &  k = 1, ..., K \
                          c_(i) & tilde.op "Categorical"(1 \/ K) quad                                   &  i = 1, ..., n \
  bold(x)_(i) | c_(i), bold(mu) & tilde.op cal(N)(c_(i)^(top)bold(mu), sigma^(2) bold(upright(I))) quad & i = 1, ..., n.
$
The joint density of observed and latent variables is
$
  p(bold(mu), bold(c), bold(x)) = p(bold(mu)) product_(i=1)^n p(c_(i)) p(bold(x)_(i) | c_(i), bold(mu)).
$


== Example: Gaussian mixture model

Construct the mean-field variational family
$
  q(bold(mu), bold(c)) = product_(k=1)^K q_k (bold(mu)_(k); bold(m)_(k), bold(upright(Sigma))_(k)) product_(i=1)^n q_i (c_(i); bold(phi)_(i)),
$
where
$
  q_k (bold(mu)_(k); bold(m)_(k), bold(upright(Sigma))_(k)) tilde.op cal(N)(bold(m)_(k), bold(upright(Sigma))_(k))
  quad"and"quad
  q_i (c_(i); bold(phi)_(i)) tilde.op "Categorical"(bold(phi)_(i)).
$
What confuses me a bit is the $bold(upright(Sigma))_k$. Do these have to be on the form $bold(upright(Sigma))_k = s_k^(2)bold(upright(I))$?

= Optimization
== Coordinate Ascent Variational Inference (CAVI)
Iteratively optimizes each factor of the mean-field variational density, while holding the others fixed.

Splitting $bold(z) = (z_(j), bold(z)_(-j))$ and $q(bold(z)) = q_(j)(z_(j))q_(-j)(bold(z)_(-j))$, and using iterated expectation, we find
$
  "ELBO"(q_(j)) prop bb(E)_(j)[bb(E)_(-j)[log p(z_(j), bold(z)_(-j), bold(x))]] - bb(E)_(j)[log q_(j)(z_(j))]
$
Maximizing this over densities $q_(j)$ gives the optimal solution (skipping the details)
$
  q_(j)^(ast)(z_(j)) prop exp{bb(E)_(-j)[log p(z_(j), bold(z)_(-j), bold(x))]}.
$

== Example: Gaussian mixture model
$
  "ELBO"(bold(m), bold(s)^(2), bold(phi)) =
  &sum_(k=1)^(K) bb(E)[log p(bold(mu)_(k)); bold(m)_(k), s_(k)^(2))] \
  &+ sum_(i=1)^(n) (bb(E)[log p(c_(i) ; phi_(i))] + bb(E)[log p(bold(x)_(i)|c_(i), bold(mu)); bold(phi)_(i), bold(m), bold(s)^(2)]) \
  &+ sum_(k=1)^(K) bb(E)[log q(bold(mu)_(k); bold(m)_(k), s_(k)^(2))] + sum_(i=1)^(n) bb(E)[log q(c_(i); phi_(i))].
$


== Example: Gaussian mixture model
#text(size: 18pt)[
  Update formula for $phi_i$
  $
    q_i^(ast)(c_(i); bold(phi)_(i))
    &prop exp{bb(E)_(bold(c)_(-i), bold(mu))[log p(c_(i), bold(c)_(-i), bold(mu), bold(x))]} \
    &prop exp{bb(E)_(bold(c)_(-i), bold(mu))[log p(c_(i)) + log p (bold(c)_(-i)) + log p(bold(mu)) + sum_(j=1)^(n) log p(x_(j)|c_(j), bold(mu))]} \
    &prop exp{bb(E)_(bold(mu))[log p(c_(i)) + log p(x_(i)|c_(i), bold(mu))]} \
    &prop exp{bb(E)_(bold(mu))[frac(1, K) + log p(bold(x)_(i)|bold(mu)_(c_(i)))]} \
    &prop exp{bb(E)_(bold(mu))[-frac(1, 2)(bold(x)_(i) - bold(mu)_(c_(i)))^(2)]} \
    &prop exp{x_(i)m_(c_(i)) - frac(1, 2)s_(c_(i))^(2) - frac(1, 2)m_(c_(i))^(2)} \
  $
]
== Example: Gaussian mixture model
#text(size: 18pt)[
  Update formulas for $m_(k)$ and $s_(k)^(2)$
  $
    q_k^(ast)(mu_(k); m_(k), s_(k)^(2))
    &prop exp{bb(E)_(bold(c), bold(mu)_(-k))[log p(bold(c), mu_(k), bold(mu)_(-k), bold(x))]} \
    &prop exp{bb(E)_(bold(c), bold(mu)_(-k))[log p (mu_(k)) + sum_(i=1)^(n) log p (x_(i)|c_(i), bold(mu))]} \
    &prop exp{bb(E)_(bold(c), bold(mu)_(-k))[-frac(1, 2 sigma^(2))mu_(k)^(2) +sum_(i=1)^(n) "I"(c_(i)=k)log p(x_(i)|mu_(k))]} \
    &prop exp{-frac(1, 2 sigma^(2))mu_(k)^(2) - frac(1, 2)sum_(i=1)^(n) phi_(i, k)(x_(i) - mu_(k))^(2)} \
    &prop exp{-frac(1, 2)(frac(1, sigma^(2)) + sum_(i=1)^(n) phi_(i, k))[mu_(k) - frac(sum_(i=1)^(n) phi_(i, k)x_(i), frac(1, sigma^(2)) + sum_(i=1)^(n) phi_(i, k))]^(2)} \
  $
]

== Black-box variational inference (BBVI)
OK

== Example: Gaussian mixture model
text

= Implementation

= A more interesting example


== Open problems
- Other distance measures that KL
- Alternatives to mean-field
  - Add dependencies between latent variables (_structured VI_)
  - Mixture of variational densities
- Interface between VI and MCMC
- Statistical properties of VI (e.g., consistency, asymptotic normality, etc.)



==
#text(size: 15pt)[
  #bibliography("refs.bib", style: "elsevier-harvard")
]

