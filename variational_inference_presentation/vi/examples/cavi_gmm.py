from vi.cavi import GMMPrior, run_cavi
from vi.data import generate_gmm_dataset
from utils import tabulate_estimates

x, centers = generate_gmm_dataset(K=3, n_samples=300, seed=42)

K = 3
prior = GMMPrior(sigma=1.0, K=K)

params = run_cavi(x, prior, n_iter=100)

tabulate_estimates(centers, params)
