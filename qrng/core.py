"""Core QRNG: build the all-Hadamard circuit, sample it on Qiskit's built-in
statevector simulator, and extract a flat bit array.

The result-extraction path here is for Qiskit 2.x / SamplerV2 and was verified
empirically (not coded from memory): a measured circuit's data lives under the
classical-register name ``meas`` (the name ``measure_all()`` assigns), and
``BitArray.to_bool_array()`` yields a ``(shots, n_qubits)`` boolean array.

The ideal simulator here is Qiskit's built-in ``StatevectorSampler`` (pure
Python/NumPy, no compiled backend). This keeps the core path free of the
``qiskit-aer`` C++ extension, which segfaults in some sandboxed hosts (e.g.
Streamlit Community Cloud). Aer is still used for *noise* modelling in
``qrng.noise``. For an all-Hadamard circuit the two give identical Born-rule
statistics.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler


def build_qrng_circuit(n_qubits: int) -> QuantumCircuit:
    """One Hadamard per qubit, then measure all -> n_qubits random bits/shot."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.measure_all()
    return qc


def sample_bitmatrix(
    n_qubits: int,
    n_shots: int,
    seed: int | None = None,
) -> np.ndarray:
    """Run the QRNG circuit on the local simulator.

    Returns a ``(n_shots, n_qubits)`` uint8 array -- one measured bitstring per
    row. Keep this 2D form for per-qubit / cross-qubit analysis; flatten it with
    :func:`generate_quantum_bits` when you just want a bit stream.
    """
    qc = build_qrng_circuit(n_qubits)
    sampler = StatevectorSampler(seed=seed)
    result = sampler.run([qc], shots=n_shots).result()
    bit_array = result[0].data["meas"]          # classical register named by measure_all()
    return bit_array.to_bool_array().astype(np.uint8)


def generate_quantum_bits(
    n_qubits: int,
    n_shots: int,
    seed: int | None = None,
) -> np.ndarray:
    """Flat 1-D uint8 array of ``n_qubits * n_shots`` quantum random bits."""
    return sample_bitmatrix(n_qubits, n_shots, seed=seed).reshape(-1)
