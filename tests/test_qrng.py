"""Unit tests for the QRNG pipeline. Run against the local simulator -- no IBM
token required. ``pytest -q`` from the repo root."""

import numpy as np
import pytest

from qrng import analysis, classical, crypto_demo, entropy_800_90b, nist


def test_secrets_bits_shape_and_values():
    bits = classical.secrets_bits(1000)
    assert bits.shape == (1000,)
    assert set(np.unique(bits)).issubset({0, 1})


def test_mt19937_reproducible_with_seed():
    a = classical.mt19937_bits(256, seed=42)
    b = classical.mt19937_bits(256, seed=42)
    assert np.array_equal(a, b)  # PRNG: deterministic given the seed


def test_mt19937_state_recovery_predicts_next_output():
    import random
    rng = random.Random(123456789)
    observed = [rng.getrandbits(32) for _ in range(624)]
    predicted = classical.mt19937_next_bit_attack(observed)
    assert predicted == rng.getrandbits(32)  # exact prediction of future output


def test_bias_report_on_balanced_stream():
    bits = classical.secrets_bits(50000)
    rep = analysis.bias_report(bits)
    assert rep.bias < 0.02
    assert rep.shannon > 0.99


def test_nist_battery_passes_on_csprng():
    bits = classical.secrets_bits(100000)
    results = nist.run_battery(bits)
    passed = sum(r.passed for r in results)
    # A CSPRNG should pass essentially all of them; allow one statistical fluke.
    assert passed >= len(results) - 1


def test_cross_qubit_chi2_independent_columns():
    rng = np.random.default_rng(0)
    shots = rng.integers(0, 2, size=(5000, 4))
    chi2 = analysis.cross_qubit_chi2(shots)
    # Independent columns -> chi2 well below the 1-dof critical value (~10.83 at p=0.001)
    assert all(v < 10.83 for v in chi2.values())


def test_aes_roundtrip_and_tamper_detection():
    bits = classical.secrets_bits(2000)
    msg = b"quantum-seeded secret"
    key, sealed = crypto_demo.encrypt(bits, msg, associated_data=b"ad")
    assert len(key) == 32  # AES-256
    assert crypto_demo.decrypt(key, sealed, associated_data=b"ad") == msg
    tampered = crypto_demo.Sealed(
        sealed.nonce, sealed.ciphertext[:-1] + bytes([sealed.ciphertext[-1] ^ 1]))
    with pytest.raises(Exception):
        crypto_demo.decrypt(key, tampered, associated_data=b"ad")


def test_800_90b_mcv_uniform_vs_biased():
    uniform = classical.secrets_bits(100000)
    rng = np.random.default_rng(0)
    biased = (rng.random(100000) < 0.7).astype(np.uint8)
    h_uniform = entropy_800_90b.mcv_estimate(uniform).min_entropy_per_bit
    h_biased = entropy_800_90b.mcv_estimate(biased).min_entropy_per_bit
    assert h_uniform > 0.95           # near-ideal source
    assert 0.45 < h_biased < 0.6      # -log2(0.7) ~ 0.515


def test_linear_complexity_fix_passes_uniform():
    # nistrng's own Linear Complexity fails uniform random (binning bug);
    # the fixed version must pass. Needs >= ~512*200 bits for stable stats.
    bits = classical.secrets_bits(512 * 250)
    from qrng.nist_full import linear_complexity_fixed
    p, passed = linear_complexity_fixed(bits)
    assert passed and p > 0.01


@pytest.mark.simulator
def test_quantum_bits_from_simulator():
    qrng_core = pytest.importorskip("qrng.core")
    bits = qrng_core.generate_quantum_bits(n_qubits=8, n_shots=2000, seed=7)
    assert bits.shape == (8 * 2000,)
    assert set(np.unique(bits)).issubset({0, 1})
    assert analysis.bias_report(bits).bias < 0.05
