"""A practical subset of the NIST SP 800-22 statistical test battery.

Each test takes a flat 0/1 ``np.ndarray`` and returns a ``TestResult`` with a
p-value; the sequence *passes* a test when ``p >= alpha`` (default 0.01). These
are necessary-but-not-sufficient checks: a good PRNG passes them all, which is
exactly why they don't by themselves prove quantum randomness.

Implemented (8 of the 15): Monobit, Block Frequency, Runs, Longest Run of Ones,
Spectral (DFT), Serial, Approximate Entropy, Cumulative Sums. These cover the
"big three" (frequency, runs, DFT) plus the pattern/entropy tests that are most
sensitive to real-hardware noise.

References: NIST SP 800-22 Rev 1a, section 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm


@dataclass
class TestResult:
    name: str
    p_value: float
    passed: bool
    detail: str = ""


def _passed(p: float, alpha: float) -> bool:
    return bool(np.isfinite(p) and p >= alpha)


def monobit(bits: np.ndarray, alpha: float = 0.01) -> TestResult:
    n = bits.size
    s = int(bits.sum()) * 2 - n  # map 0/1 -> -1/+1 and sum
    s_obs = abs(s) / np.sqrt(n)
    p = erfc(s_obs / np.sqrt(2))
    return TestResult("Frequency (Monobit)", float(p), _passed(p, alpha))


def block_frequency(bits: np.ndarray, m: int = 128, alpha: float = 0.01) -> TestResult:
    n = bits.size
    nblocks = n // m
    if nblocks == 0:
        return TestResult("Block Frequency", float("nan"), False, "too few bits")
    blocks = bits[: nblocks * m].reshape(nblocks, m)
    pi = blocks.mean(axis=1)
    chi2 = 4 * m * np.sum((pi - 0.5) ** 2)
    p = gammaincc(nblocks / 2, chi2 / 2)
    return TestResult("Block Frequency", float(p), _passed(p, alpha), f"M={m}, N={nblocks}")


def runs(bits: np.ndarray, alpha: float = 0.01) -> TestResult:
    n = bits.size
    pi = bits.mean()
    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return TestResult("Runs", 0.0, False, "failed monobit pre-condition")
    vobs = 1 + int(np.sum(bits[1:] != bits[:-1]))
    num = abs(vobs - 2 * n * pi * (1 - pi))
    den = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    p = erfc(num / den)
    return TestResult("Runs", float(p), _passed(p, alpha))


def longest_run_of_ones(bits: np.ndarray, alpha: float = 0.01) -> TestResult:
    n = bits.size
    # Parameter sets from SP 800-22 table for the three size regimes.
    if n < 128:
        return TestResult("Longest Run of Ones", float("nan"), False, "need n>=128")
    if n < 6272:
        m, k, nblocks = 8, 3, 16
        v_classes = [1, 2, 3, 4]                    # <=1, 2, 3, >=4
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        m, k, nblocks = 128, 5, 49
        v_classes = [4, 5, 6, 7, 8, 9]              # <=4 ... >=9
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        m, k, nblocks = 10000, 6, 75
        v_classes = [10, 11, 12, 13, 14, 15, 16]
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    blocks = bits[: nblocks * m].reshape(nblocks, m)
    counts = np.zeros(len(v_classes), dtype=int)
    for blk in blocks:
        # longest run of ones in this block
        longest = best = 0
        for b in blk:
            best = best + 1 if b else 0
            longest = max(longest, best)
        lo, hi = v_classes[0], v_classes[-1]
        idx = min(max(longest, lo), hi) - lo
        counts[idx] += 1
    expected = nblocks * np.array(pi)
    chi2 = np.sum((counts - expected) ** 2 / expected)
    p = gammaincc(k / 2, chi2 / 2)
    return TestResult("Longest Run of Ones", float(p), _passed(p, alpha), f"M={m}")


def spectral_dft(bits: np.ndarray, alpha: float = 0.01) -> TestResult:
    n = bits.size
    x = bits.astype(np.float64) * 2 - 1
    mags = np.abs(np.fft.fft(x))[: n // 2]
    t = np.sqrt(np.log(1 / 0.05) * n)
    n0 = 0.95 * n / 2
    n1 = int(np.sum(mags < t))
    d = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4)
    p = erfc(abs(d) / np.sqrt(2))
    return TestResult("Spectral (DFT)", float(p), _passed(p, alpha))


def _pattern_counts(bits: np.ndarray, m: int) -> np.ndarray:
    """Counts of every overlapping m-bit pattern on the circularly-extended
    sequence, vectorised via a sliding-window view (no Python loop)."""
    n = bits.size
    ext = np.concatenate([bits, bits[: m - 1]]).astype(np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(ext, m)  # (n, m)
    weights = (1 << np.arange(m)[::-1]).astype(np.int64)
    idx = windows @ weights
    return np.bincount(idx, minlength=1 << m)


def _psi2(bits: np.ndarray, m: int) -> float:
    n = bits.size
    if m <= 0:
        return 0.0
    counts = _pattern_counts(bits, m)
    return (counts.astype(np.float64) ** 2).sum() * (1 << m) / n - n


def serial(bits: np.ndarray, m: int = 10, alpha: float = 0.01) -> TestResult:
    # NIST validity bound: m < floor(log2 n) - 2. Cap so the 2^m pattern space
    # is actually populated; m too large makes the p-value meaningless.
    m_max = int(np.floor(np.log2(bits.size))) - 3
    m = max(2, min(m, m_max))
    psi_m = _psi2(bits, m)
    psi_m1 = _psi2(bits, m - 1)
    psi_m2 = _psi2(bits, m - 2)
    d1 = psi_m - psi_m1
    d2 = psi_m - 2 * psi_m1 + psi_m2
    p1 = gammaincc(2 ** (m - 2), d1 / 2)
    p2 = gammaincc(2 ** (m - 3), d2 / 2)
    # report the more demanding (smaller) p-value
    p = min(p1, p2)
    return TestResult("Serial", float(p), _passed(p, alpha), f"m={m}, p1={p1:.3f}, p2={p2:.3f}")


def approximate_entropy(bits: np.ndarray, m: int = 8, alpha: float = 0.01) -> TestResult:
    n = bits.size
    # NIST guidance: m < floor(log2 n) - 5. Cap so phi(m+1) is well estimated.
    m_max = int(np.floor(np.log2(n))) - 6
    m = max(2, min(m, m_max))

    def phi(mm: int) -> float:
        if mm == 0:
            return 0.0
        c = _pattern_counts(bits, mm).astype(np.float64) / n
        nz = c[c > 0]
        return float(np.sum(nz * np.log(nz)))

    apen = phi(m) - phi(m + 1)
    chi2 = 2 * n * (np.log(2) - apen)
    p = gammaincc(2 ** (m - 1), chi2 / 2)
    return TestResult("Approximate Entropy", float(p), _passed(p, alpha), f"m={m}")


def cumulative_sums(bits: np.ndarray, alpha: float = 0.01) -> TestResult:
    n = bits.size
    x = bits.astype(np.int64) * 2 - 1
    fwd = np.max(np.abs(np.cumsum(x)))
    bwd = np.max(np.abs(np.cumsum(x[::-1])))
    z = max(fwd, bwd)

    def _p(z_val: float) -> float:
        sqn = np.sqrt(n)
        k1 = np.arange(int((-n / z_val + 1) / 4), int((n / z_val - 1) / 4) + 1)
        term1 = np.sum(norm.cdf((4 * k1 + 1) * z_val / sqn) - norm.cdf((4 * k1 - 1) * z_val / sqn))
        k2 = np.arange(int((-n / z_val - 3) / 4), int((n / z_val - 1) / 4) + 1)
        term2 = np.sum(norm.cdf((4 * k2 + 3) * z_val / sqn) - norm.cdf((4 * k2 + 1) * z_val / sqn))
        return float(1 - term1 + term2)

    p = _p(z) if z > 0 else 1.0
    return TestResult("Cumulative Sums", float(p), _passed(p, alpha))


ALL_TESTS = [
    monobit,
    block_frequency,
    runs,
    longest_run_of_ones,
    spectral_dft,
    serial,
    approximate_entropy,
    cumulative_sums,
]


def run_battery(bits: np.ndarray, alpha: float = 0.01) -> list[TestResult]:
    """Run the full implemented battery; returns one TestResult per test."""
    bits = np.asarray(bits, dtype=np.uint8)
    results = []
    for test in ALL_TESTS:
        try:
            results.append(test(bits, alpha=alpha))
        except Exception as exc:  # keep the battery going if one test errors
            results.append(TestResult(test.__name__, float("nan"), False, f"error: {exc}"))
    return results
