from typing import NamedTuple


class GMMPrior(NamedTuple):
    sigma: float = 1.0
    K: int = 3
