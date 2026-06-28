"""Output-based randomness diagnostics: bias, Shannon entropy, min-entropy,
and inter-bit / cross-qubit correlation.

These operate on a flat ``np.ndarray`` of 0/1 bits regardless of source
(quantum, MT19937, secrets), so the same numbers are directly comparable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class BiasReport:
    n: int
    ones: int
    zeros: int
    p1: float
    bias: float          # |p1 - 0.5|
    shannon: float       # bits per bit, ideal = 1.0
    min_entropy: float   # per-bit min-entropy, ideal = 1.0


def bias_report(bits: np.ndarray) -> BiasReport:
    """Global 0/1 balance, Shannon entropy and single-bit min-entropy."""
    bits = np.asarray(bits)
    n = bits.size
    ones = int(bits.sum())
    zeros = n - ones
    p1 = ones / n if n else 0.0
    p0 = 1.0 - p1

    def _h(p: float) -> float:
        return 0.0 if p in (0.0, 1.0) else -p * np.log2(p)

    shannon = _h(p0) + _h(p1)
    pmax = max(p0, p1)
    min_ent = -np.log2(pmax) if pmax > 0 else 0.0
    return BiasReport(n, ones, zeros, p1, abs(p1 - 0.5), float(shannon), float(min_ent))


def block_min_entropy(bits: np.ndarray, block_size: int = 8) -> float:
    """Per-bit min-entropy estimated over ``block_size``-bit symbols.

    H_inf = -log2(max_x P(X=x)), normalised per bit. For an ideal source this
    approaches 1.0 bit/bit. NOTE: this is an *output* statistic -- a seeded
    MT19937 stream scores ~1.0 here too, because the estimator cannot see the
    seed. The PRNG/TRNG distinction is predictability-given-state, not this
    number (see classical.mt19937_next_bit_attack).
    """
    bits = np.asarray(bits)
    n_blocks = bits.size // block_size
    if n_blocks == 0:
        raise ValueError("not enough bits for one block")
    trimmed = bits[: n_blocks * block_size].reshape(n_blocks, block_size)
    # pack each row of bits into an integer symbol
    weights = (1 << np.arange(block_size)[::-1]).astype(np.int64)
    symbols = trimmed.astype(np.int64) @ weights
    counts = Counter(symbols.tolist())
    pmax = max(counts.values()) / n_blocks
    return float(-np.log2(pmax) / block_size)


def serial_correlation(bits: np.ndarray, lag: int = 1) -> float:
    """Lag-``k`` autocorrelation of the bit stream; ideal ~ 0."""
    bits = np.asarray(bits, dtype=np.float64)
    if bits.size <= lag:
        return float("nan")
    a = bits[:-lag] - bits.mean()
    b = bits[lag:] - bits.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else 0.0


def cross_qubit_chi2(per_qubit_bits: np.ndarray) -> dict[tuple[int, int], float]:
    """Chi-squared independence test between neighbouring qubit columns.

    ``per_qubit_bits`` is shape (n_shots, n_qubits): each row is one shot's
    measured bitstring. Returns {(i, i+1): chi2} for adjacent qubit pairs.
    Crosstalk on real hardware shows up as departures from independence here.
    """
    arr = np.asarray(per_qubit_bits)
    n_shots, n_qubits = arr.shape
    out: dict[tuple[int, int], float] = {}
    for i in range(n_qubits - 1):
        a, b = arr[:, i], arr[:, i + 1]
        # 2x2 contingency table
        obs = np.array([
            [np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))],
            [np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))],
        ], dtype=np.float64)
        row = obs.sum(1, keepdims=True)
        col = obs.sum(0, keepdims=True)
        exp = row @ col / n_shots
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0).sum()
        out[(i, i + 1)] = float(chi2)
    return out
