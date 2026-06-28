"""NIST SP 800-90B min-entropy estimators (a subset).

SP 800-90B assesses an *entropy source* (unlike SP 800-22, which only tests an
output bit stream). Its full non-IID track takes the minimum over ten estimators;
we implement two of the canonical ones and label results accordingly -- this is
NOT "the 800-90B suite", it is two named estimators from it.

  * MCV (Most Common Value), SP 800-90B section 6.3.1 -- the IID estimator. Uses
    the upper confidence bound on the most-common-value probability, so it is
    conservative and self-validating (uniform -> ~1.0, biased -> lower).
  * Markov, section 6.3.3 -- a non-IID estimator that accounts for first-order
    dependence between successive bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MinEntropyEstimate:
    estimator: str
    min_entropy_per_bit: float
    detail: str = ""


def mcv_estimate(bits: np.ndarray, alpha: float = 0.99) -> MinEntropyEstimate:
    """Most-Common-Value min-entropy (SP 800-90B 6.3.1), bits as the alphabet.

    p_hat = max symbol frequency; upper bound p_u = p_hat + z * sqrt(p_hat(1-p_hat)/(n-1)),
    H_min = -log2(p_u). z = 2.576 for the 99% one-sided bound NIST specifies.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    ones = int(bits.sum())
    p_hat = max(ones, n - ones) / n
    z = 2.5758293035489008  # 99% one-sided normal quantile
    p_u = min(1.0, p_hat + z * np.sqrt(p_hat * (1 - p_hat) / (n - 1)))
    h = -np.log2(p_u)
    return MinEntropyEstimate("MCV (IID, 800-90B 6.3.1)", float(h),
                              f"p_hat={p_hat:.5f}, p_upper={p_u:.5f}")


def markov_estimate(bits: np.ndarray) -> MinEntropyEstimate:
    """First-order Markov min-entropy (SP 800-90B 6.3.3), binary alphabet.

    Estimates initial and transition probabilities, then bounds the min-entropy
    of the most likely length-128 path and normalises per bit (per the standard's
    construction for the binary case).
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    p0 = float((bits == 0).mean())
    p1 = 1.0 - p0
    # transition counts
    a, b = bits[:-1], bits[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    p00 = n00 / max(n00 + n01, 1)
    p01 = n01 / max(n00 + n01, 1)
    p10 = n10 / max(n10 + n11, 1)
    p11 = n11 / max(n10 + n11, 1)

    # SP 800-90B Markov: most-likely path probability over L=128 steps, then
    # H_min = min(-log2(p_path)/128, 1). Take the max-probability initial state
    # and greedily the max transition, as the standard's binary construction does.
    import math
    L = 128
    candidates = []
    for init, p_init in ((0, p0), (1, p1)):
        # worst case: follow the most probable transitions
        logp = math.log2(p_init) if p_init > 0 else -math.inf
        for _ in range(L - 1):
            if init == 0:
                step = max(p00, p01); init = 0 if p00 >= p01 else 1
            else:
                step = max(p10, p11); init = 0 if p10 >= p11 else 1
            logp += math.log2(step) if step > 0 else -math.inf
        candidates.append(-logp)
    h = min(min(candidates) / L, 1.0)
    return MinEntropyEstimate("Markov (non-IID, 800-90B 6.3.3)", float(h),
                              f"p00={p00:.4f} p11={p11:.4f}")


def assess(bits: np.ndarray) -> list[MinEntropyEstimate]:
    """Run the implemented estimators; the conservative source estimate is the
    minimum of them (as the non-IID track does over its full set)."""
    ests = [mcv_estimate(bits), markov_estimate(bits)]
    return ests
