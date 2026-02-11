"""Variational Inference library (JAX)."""

from vi.bbvi import BBVIResult, run_bbvi
from vi.cavi import CAVIPosterior, NormalGammaPrior, run_cavi

__all__ = [
    "BBVIResult",
    "CAVIPosterior",
    "NormalGammaPrior",
    "run_bbvi",
    "run_cavi",
]
