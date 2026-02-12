"""Variational Inference library (JAX)."""

from vi.bbvi import run_bbvi
from vi.cavi import run_cavi
from vi.gmm import GMMPrior

__all__ = [
    "GMMPrior",
    "run_bbvi",
    "run_cavi",
]
