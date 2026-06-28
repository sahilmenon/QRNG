"""Error mitigation and randomness post-processing.

Two distinct things, often conflated:

  * **Readout mitigation** estimates the per-qubit confusion matrix and inverts
    it to recover the *true* 0/1 probabilities from biased measured counts. This
    corrects a *statistic* (the bias estimate); it cannot un-flip an individual
    measured bit.

  * **Randomness extraction** (Von Neumann) turns a biased-but-independent bit
    stream into an unbiased one, provably driving min-entropy toward 1.0 bit/bit
    at the cost of throughput. This is the practical way a QRNG "recovers"
    near-ideal entropy after noisy measurement.
"""

from __future__ import annotations

import numpy as np


def von_neumann_extract(bits: np.ndarray) -> np.ndarray:
    """Von Neumann debiasing: map bit pairs 01->0, 10->1, discard 00/11.

    For i.i.d. bits with any fixed bias p, the output is exactly unbiased
    (P(01)=P(10)=p(1-p)). Throughput is ~p(1-p) of the input (<= 25%). Serial
    correlation in the input weakens the guarantee, so report the input's
    autocorrelation alongside.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n_pairs = bits.size // 2
    pairs = bits[: n_pairs * 2].reshape(n_pairs, 2)
    a, b = pairs[:, 0], pairs[:, 1]
    keep = a != b
    return a[keep].astype(np.uint8)


def readout_confusion_matrices(n_qubits: int, n_shots: int = 4000,
                               noise_model=None, seed: int | None = 1) -> np.ndarray:
    """Calibrate per-qubit 2x2 readout confusion matrices on the simulator.

    Prepares all-|0> and all-|1> states, measures, and estimates
    M[q] = [[P(0|0), P(1|0)], [P(0|1), P(1|1)]] for each qubit. Returns an
    array of shape ``(n_qubits, 2, 2)``.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer.primitives import SamplerV2

    opts = {"backend_options": {"noise_model": noise_model}} if noise_model else {}
    sampler = SamplerV2(seed=seed, options=opts)

    def _measure(prepare_ones: bool) -> np.ndarray:
        qc = QuantumCircuit(n_qubits)
        if prepare_ones:
            qc.x(range(n_qubits))
        qc.measure_all()
        res = sampler.run([qc], shots=n_shots).result()
        return res[0].data["meas"].to_bool_array().astype(np.uint8)

    zeros = _measure(False)   # prepared |0>
    ones = _measure(True)     # prepared |1>
    mats = np.zeros((n_qubits, 2, 2))
    for q in range(n_qubits):
        p1_given0 = zeros[:, q].mean()
        p1_given1 = ones[:, q].mean()
        mats[q] = [[1 - p1_given0, p1_given0], [1 - p1_given1, p1_given1]]
    return mats


def mitigate_bias_estimate(measured_p1: float, confusion: np.ndarray) -> float:
    """Invert a single qubit's confusion matrix to recover the true P(1).

    measured = M^T @ true, with measured = [P_meas(0), P_meas(1)]. Solve for the
    true distribution and clip to [0, 1].
    """
    m = np.array([1 - measured_p1, measured_p1])
    true = np.linalg.solve(confusion.T, m)
    return float(np.clip(true[1], 0.0, 1.0))
