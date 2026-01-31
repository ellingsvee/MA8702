data {
  int<lower=1> N;          // number of pumps
  int<lower=0> y[N];       // failures
  vector<lower=0>[N] t;    // operating times
}

parameters {
  real<lower=0> alpha;
  real<lower=0> beta;
  vector<lower=0>[N] lambda;
}

model {
  // hyperpriors
  alpha ~ exponential(1.0);
  beta  ~ gamma(0.1, 1.0);

  // prior for individual failure rates
  lambda ~ gamma(alpha, beta);

  // likelihood
  y ~ poisson(lambda .* t);
}
