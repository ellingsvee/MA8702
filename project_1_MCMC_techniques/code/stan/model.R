library(rstan)
library(bayesplot)
library(ggplot2)

# Allow parallel chains
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

y <- c(5, 1, 5, 14, 3, 19, 1, 4, 22, 10)
t <- c(94.3, 15.7, 62.9, 126.0, 5.24, 31.4, 1.05, 1.05, 2.12, 10.5)

stan_data <- list(
  N = length(y),
  y = y,
  t = t
)

fit <- stan(
  file = "pump.stan",  # make sure "pump.stan" is in your working dir
  data = stan_data,
  chains = 4,
  iter = 10000,
  warmup = 2000,
  seed = 123
)

# Print summary of parameters including ESS and R-hat
print(fit, pars = c("alpha", "beta", "lambda"))

# Convert to array for bayesplot
posterior_array <- as.array(fit)

# Traceplots
trace_svg_file <- "traceplots.svg"
svg(trace_svg_file, width = 10, height = 6)
mcmc_trace(posterior_array, pars = c("alpha", "beta"))
dev.off()

# Traceplots for individual lambda_i
trace_lambda_svg <- "trace_lambda.svg"
svg(trace_lambda_svg, width = 10, height = 8)
mcmc_trace(posterior_array, pars = paste0("lambda[", 1:10, "]"))
dev.off()

area_svg <- "posterior_area.svg"
svg(area_svg, width = 10, height = 6)
mcmc_areas(
  posterior_array,
  pars = c("alpha", "beta"),
  prob = 0.8,
  prob_outer = 0.95
)
dev.off()

# Individual lambda_i intervals
interval_svg <- "lambda_intervals.svg"
svg(interval_svg, width = 10, height = 8)
mcmc_intervals(
  posterior_array,
  pars = paste0("lambda[", 1:10, "]"),
  prob = 0.8,
  prob_outer = 0.95
)
dev.off()

