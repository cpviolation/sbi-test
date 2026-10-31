"""Pure-NumPy phase-space helpers: DecaytimePhaseSpace and CombinedPhaseSpace.

Adapted from (TensorFlow originals):
  DecaytimePhaseSpace:
    https://github.com/cpviolation/tfa_examples/blob/main/develtfa/decaytime_phasespace.py
  CombinedPhaseSpace:
    https://github.com/cpviolation/AmpliTF/blob/master/amplitf/phasespace/combined_phasespace.py
"""

import numpy as np


class DecaytimePhaseSpace:
    """1-D decay-time phase space.

    Generates decay times t drawn from an exponential distribution with
    lifetime ``tau`` (t > 0).
    """

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    # ------------------------------------------------------------------
    def dimensionality(self) -> int:
        return 1

    def bounds(self):
        return [(0.0, float("inf"))]

    # ------------------------------------------------------------------
    def inside(self, x: np.ndarray) -> np.ndarray:
        """Boolean mask: True where t > 0.

        ``x`` may be shape (N,) or (N, 1); returns shape (N,).
        """
        t = x[:, 0] if x.ndim == 2 else x
        return t > 0.0

    def filter(self, x: np.ndarray) -> np.ndarray:
        return x[self.inside(x)]

    # ------------------------------------------------------------------
    def unfiltered_sample(self, size: int, maximum=None) -> np.ndarray:
        """Generate *size* points; returns shape (size, 1) [or (size, 2) with majorant].

        Decay times are drawn via inverse-CDF: t = -tau * ln(1-U), U~Uniform(0,1).
        If ``maximum`` is given, an extra uniform(0, maximum) column is appended
        (classic majorant column for accept-reject MC).
        """
        u = np.random.uniform(0.0, 1.0, size)
        t = -self.tau * np.log1p(-u)           # numerically stable
        cols = [t[:, np.newaxis]]
        if maximum is not None:
            cols.append(np.random.uniform(0.0, maximum, (size, 1)))
        return np.concatenate(cols, axis=1)

    def uniform_sample(self, size: int, maximum=None) -> np.ndarray:
        """Filtered sample — all returned points satisfy t > 0 (always true here)."""
        return self.filter(self.unfiltered_sample(size, maximum))

    # ------------------------------------------------------------------
    def t(self, sample: np.ndarray) -> np.ndarray:
        """Extract the decay-time column; returns shape (N,)."""
        return sample[:, 0] if sample.ndim == 2 else sample


# ======================================================================

class CombinedPhaseSpace:
    """Cartesian (direct) product of two phase spaces.

    ``phsp1`` occupies the first ``phsp1.dimensionality()`` columns;
    ``phsp2`` occupies the remaining columns.
    """

    def __init__(self, phsp1, phsp2):
        self.phsp1 = phsp1
        self.phsp2 = phsp2

    # ------------------------------------------------------------------
    def dimensionality(self) -> int:
        return self.phsp1.dimensionality() + self.phsp2.dimensionality()

    def bounds(self):
        return self.phsp1.bounds() + self.phsp2.bounds()

    # ------------------------------------------------------------------
    def data1(self, x: np.ndarray) -> np.ndarray:
        """First phase-space slice (phsp1 columns)."""
        return x[..., : self.phsp1.dimensionality()]

    def data2(self, x: np.ndarray) -> np.ndarray:
        """Second phase-space slice (phsp2 columns)."""
        return x[..., self.phsp1.dimensionality():]

    # ------------------------------------------------------------------
    def inside(self, x: np.ndarray) -> np.ndarray:
        return self.phsp1.inside(self.data1(x)) & self.phsp2.inside(self.data2(x))

    def filter(self, x: np.ndarray) -> np.ndarray:
        return x[self.inside(x)]

    # ------------------------------------------------------------------
    def unfiltered_sample(self, size: int, maximum=None) -> np.ndarray:
        """Generate *size* combined points.

        ``maximum`` is forwarded to ``phsp2`` (majorant column for accept-reject),
        consistently with the TF original.
        """
        s1 = self.phsp1.unfiltered_sample(size)          # (size, d1)
        s2 = self.phsp2.unfiltered_sample(size, maximum) # (size, d2[+1])
        if s1.ndim == 1:
            s1 = s1[:, np.newaxis]
        if s2.ndim == 1:
            s2 = s2[:, np.newaxis]
        return np.concatenate([s1, s2], axis=1)

    def uniform_sample(self, size: int, maximum=None) -> np.ndarray:
        return self.filter(self.unfiltered_sample(size, maximum))
