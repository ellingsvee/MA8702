"""Variational Inference library (JAX)."""

from vi.bbvi import BBVIResult, run_bbvi
from vi.cavi import CAVIPosterior, GMMPrior, run_cavi

__all__ = [
    "BBVIResult",
    "CAVIPosterior",
    "GMMPrior",
    "run_bbvi",
    "run_cavi",
]
