"""Noise characterisation on the simulator: build representative hardware-like
noise models, and sweep circuit depth to expose decoherence (T1/T2) effects.

On a real device, extra idle time before measurement lets qubits relax toward
|0>, so a superposition that should read 50/50 drifts. We reproduce that here by
inserting ``depth`` identity gates (each carrying a thermal-relaxation error)
between the Hadamards and the measurement, then measuring entropy vs depth.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .analysis import bias_report

# NOTE: qiskit-aer is imported lazily inside each function, not at module top.
# Its C++ extension segfaults in some sandboxed hosts (e.g. Streamlit Community
# Cloud), so merely *importing* this module must not load it. Callers that use
# noise models still need qiskit-aer installed and a host that can run it.


def readout_noise_model(p0_given1: float = 0.03, p1_given0: float = 0.02) -> "NoiseModel":
    """Asymmetric readout error only -- the dominant bias source on real QPUs."""
    from qiskit_aer.noise import NoiseModel, ReadoutError
    nm = NoiseModel()
    # rows: P(read | prepared);  [[P(0|0),P(1|0)],[P(0|1),P(1|1)]]
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - p1_given0, p1_given0], [p0_given1, 1 - p0_given1]])
    )
    return nm


def thermal_noise_model(
    t1_us: float = 100.0,
    t2_us: float = 80.0,
    gate_ns: float = 100.0,
    readout: bool = True,
) -> "NoiseModel":
    """Thermal-relaxation error on idle/single-qubit gates (+ optional readout).

    Times in microseconds; gate duration in nanoseconds. Defaults are in the
    ballpark of current superconducting hardware.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, thermal_relaxation_error
    nm = NoiseModel()
    t1, t2, gate = t1_us * 1e3, t2_us * 1e3, gate_ns  # all -> ns
    relax = thermal_relaxation_error(t1, t2, gate)
    nm.add_all_qubit_quantum_error(relax, ["id", "h", "x", "sx", "rz"])
    if readout:
        nm.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.03, 0.97]]))
    return nm


def _sampler(noise_model, seed: int | None):
    from qiskit_aer.primitives import SamplerV2
    opts = {"backend_options": {"noise_model": noise_model}} if noise_model else {}
    return SamplerV2(seed=seed, options=opts)


def sample_with_noise(
    n_qubits: int,
    n_shots: int,
    noise_model: NoiseModel | None = None,
    depth: int = 0,
    seed: int | None = None,
) -> np.ndarray:
    """All-Hadamard circuit, then ``depth`` identity gates per qubit, measured.

    ``depth`` idle gates accumulate relaxation error without changing the ideal
    (uniform) distribution, isolating the decoherence effect. Returns a
    ``(n_shots, n_qubits)`` uint8 array.
    """
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for _ in range(depth):
        qc.id(range(n_qubits))
    qc.measure_all()
    result = _sampler(noise_model, seed).run([qc], shots=n_shots).result()
    return result[0].data["meas"].to_bool_array().astype(np.uint8)


def depth_sweep(
    n_qubits: int = 5,
    n_shots: int = 4000,
    depths: list[int] | None = None,
    noise_model: NoiseModel | None = None,
    seed: int | None = 42,
) -> dict[str, list[float]]:
    """Measure Shannon entropy and bias as a function of inserted idle depth.

    With a thermal noise model, entropy should fall and bias should grow as
    depth increases -- the decoherence signature.
    """
    if depths is None:
        depths = [0, 5, 10, 20, 40, 80, 160]
    if noise_model is None:
        noise_model = thermal_noise_model()
    out = {"depth": [], "entropy": [], "bias": []}
    for d in depths:
        mat = sample_with_noise(n_qubits, n_shots, noise_model, depth=d, seed=seed)
        rep = bias_report(mat.reshape(-1))
        out["depth"].append(d)
        out["entropy"].append(rep.shannon)
        out["bias"].append(rep.bias)
    return out
