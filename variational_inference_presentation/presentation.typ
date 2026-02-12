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
  size: 25pt,
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
Variational inference handles this problem through optimization
$
  q^(ast)(bold(z)) = "argmin"_(q(bold(z)) in cal(D)) "KL"(q(bold(z)) || p(bold(z)|bold(x))).
$
Here, $cal(D)$ is a family of densities over $bold(z)$, and $"KL"(dot)$ denotes the Kullback-Leibler divergence
$
  "KL"(q(bold(z)) || p(bold(z)|bold(x))) = bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))] - bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z)|bold(x))].
$
However, we still do not know how to compute $p(bold(z)|bold(x))$.

== Evidence Lower Bound (ELBO)
$
  "KL"(q(bold(z)) || p(bold(z)|bold(x)))
  &= bb(E)_(bold(z) tilde.op bold(q))[log q(bold(z))] - bb(E)_(bold(z) tilde.op bold(q))[log p(bold(z)|bold(x))].
$

== Mean-field variational family

Have to determine the family $cal(D)$ of approximate densities. Assume that the latent variables are mutually independent and each governed by a distinct factor in the variational density.

$
  q(bold(z)) = product_(j=1)^m q_j (bold(z)_j).
$
_This trick also ensures a diagonal covariance matrix..._

== Example: Gaussian mixture model
$K$ clusters, $n$ observations, and $d$-dimensional data.
// Latent variables $bold(z) = (bold(mu)^top, bold(c)^top)^(top)$.
$
                   bold(mu)_(k) & tilde.op cal(N)(bold(0), sigma^(2) bold(upright(I))) quad             &  k = 1, ..., K \
                          c_(i) & tilde.op "Categorical"(1 \/ K) quad                                   &  i = 1, ..., n \
  bold(x)_(i) | c_(i), bold(mu) & tilde.op cal(N)(c_(i)^(top)bold(mu), sigma^(2) bold(upright(I))) quad & i = 1, ..., n.
$
Here, the latent variables are $bold(z) = (bold(mu), bold(c))$.




//
//
//
// == Problem
// Consider a joint density of latent variables $bold(z)$ and observations $bold(x)$
// $
//   p(bold(z), bold(x)) = p(bold(z)) p(bold(x)|bold(z)).
// $
// A Bayesian model draws the latent variables from a prior density $p(bold(z))$ and then relates them to the observations through the likelihood $p(bold(x)|bold(z))$. Inference in a Bayesian model amounts to  conditioning on data and computing the posterior $p(bold(z)|bold(x))$.
//
// Issues:
// - Slow for large datasets of complex models
//
// == Variational inference
// Use optimization instead of sampling!
//
// Choose a _family_ of approximate densities $cal(D)$, which is a set of densities over the latent variables. Try to find a member that minimizes the Kullback-Leibler divergence to the exact posterior
// $
//   q^(ast)(bold(z)) = "argmin"_(q(bold(z)) in cal(D)) "KL"(q(bold(z)) || p(bold(z)|bold(x))).
// $
//
// Need to choose $cal(D)$ to be flexible enough to capture a density close to $p(bold(z)|bold(x))$, but simple enough for efficient optimization.
//
// Connection to the EM algorithm
//
// == Benefits and disadvantages
//
// Benefits:
// + Can be better that general MCMC for mixture models with multiple modes.
//
// Disadvantages:
// + No theoretical guarantees that we produce exactly the true posterior.
//
// _Variational inference is a valuable tool, alongside MCMC, in the statistician’s toolbox @blei_variational_2017 _
//
// == Evidence lower bound
// $
//   q^(ast)(bold(z)) & = "argmin"_(q(bold(z)) in cal(D)) "KL"(q(bold(z)) || p(bold(z)|bold(x))) \
//                    & = "argmin"_(q(bold(z)) in cal(D)) [bb(E)(log q(bold(z))) - bb(E)(log p(bold(z)|bold(x)))].
// $
// is intractable because of the dependence on $p(bold(x))$.
//
// Instead optimize over objective function called the evidence lower bound (ELBO):
// $
//   "ELBO"(q) = bb(E)[log p(bold(z), bold(x))] - bb(E)[log q(bold(z))].
// $
// Note the relation $log p(bold(x)) = "KL"(q(bold(z))||p(bold(z)|bold(x))) + "ELBO"(q)$.
//
// == Mean-field variational family
//
// $
//   q(bold(z)) = product_(j=1)^m q_j (bold(z)_j).
// $
// where each latent variable $bold(z)_j$ is governed by its own variational factor, the density $q_j (bold(z)_(j))$
//
//

==
#text(size: 15pt)[
  #bibliography("refs.bib", style: "elsevier-harvard")
]

