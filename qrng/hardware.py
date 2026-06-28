"""Submit the QRNG circuit to real IBM Quantum hardware.

This module is import-safe without credentials; it only touches the network when
you call :func:`generate_quantum_bits_hardware`. Save your token once with::

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",   # NOTE: legacy "ibm_quantum" was removed
        token="<API_KEY>",
        instance="<INSTANCE_CRN>",        # from the IBM Quantum Platform dashboard
        overwrite=True,
    )

The IBM Quantum Platform migrated to IBM Cloud; the old ``channel="ibm_quantum"``
no longer exists (valid: ``ibm_quantum_platform``, ``ibm_cloud``). Nothing here
hardcodes a backend name or free-tier quota -- both drift over time. The backend
is resolved at submission via ``least_busy`` and printed so the run is
reproducible after the fact.
"""

from __future__ import annotations

import numpy as np

from .core import build_qrng_circuit


def generate_quantum_bits_hardware(
    n_qubits: int,
    n_shots: int,
    backend_name: str | None = None,
    return_matrix: bool = False,
):
    """Run the all-Hadamard circuit on an IBM QPU and return the measured bits.

    Lazily imports ``qiskit_ibm_runtime`` so the rest of the project works
    without it installed/configured. Returns a flat uint8 bit array, or a
    ``(n_shots, n_qubits)`` matrix when ``return_matrix=True``.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler

    service = QiskitRuntimeService()
    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False)
    print(f"[hardware] using backend: {backend.name}")

    qc = build_qrng_circuit(n_qubits)
    # Pin the layout so the cross-qubit correlation test reflects *physical*
    # adjacency. Without this the transpiler is free to remap virtual->physical
    # qubits and the "neighbouring qubit" crosstalk claim no longer holds.
    initial_layout = list(range(n_qubits))
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, initial_layout=initial_layout)
    qc_t = pm.run(qc)   # produce an ISA circuit for this backend
    print(f"[hardware] physical qubits used (initial_layout): {initial_layout}")

    sampler = RuntimeSampler(mode=backend)
    job = sampler.run([qc_t], shots=n_shots)
    print(f"[hardware] job id: {job.job_id()} -- queued, this can take minutes")
    result = job.result()

    # Runtime SamplerV2 result extraction mirrors the Aer path: the register
    # carries the name measure_all() assigned. Resolve it defensively.
    data = result[0].data
    reg_name = next(iter(data.keys())) if hasattr(data, "keys") else "meas"
    bit_array = data[reg_name]
    matrix = bit_array.to_bool_array().astype(np.uint8)
    return matrix if return_matrix else matrix.reshape(-1)
