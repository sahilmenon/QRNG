"""Streamlit demo for the Quantum Random Number Generator.

Runs entirely on the local Aer simulator, so it deploys to a free public host
(Streamlit Community Cloud / Hugging Face Spaces) with no IBM token or secrets.

Local:   streamlit run app.py
Deploy:  push to GitHub -> share.streamlit.io -> pick this repo -> app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from qrng import (
    analysis, classical, core, crypto_demo, entropy_800_90b, mitigation, nist, noise,
)

st.set_page_config(page_title="Quantum RNG", page_icon="🎲", layout="wide")

st.title("🎲 Quantum Random Number Generator")
st.caption(
    "Hadamard gates collapse qubit superpositions into provably random bits, "
    "then we hold them to the same statistical scrutiny as classical RNGs. "
    "Runs on Qiskit's Aer simulator (no IBM token needed for this demo)."
)

with st.sidebar:
    st.header("Parameters")
    n_qubits = st.slider("Qubits", 2, 16, 8)
    n_shots = st.select_slider("Shots", [1000, 2500, 5000, 12500, 25000], value=12500)
    apply_noise = st.checkbox("Hardware-like noise model", value=False)
    seed = st.number_input("Seed", value=2025, step=1)
    go = st.button("Generate quantum bits", type="primary")
    st.markdown("---")
    st.caption(f"Will generate **{n_qubits * n_shots:,}** bits "
               f"({n_qubits} qubits × {n_shots:,} shots).")


def summarise(name: str, bits: np.ndarray) -> dict:
    rep = analysis.bias_report(bits)
    battery = nist.run_battery(bits)
    mcv = entropy_800_90b.mcv_estimate(bits).min_entropy_per_bit
    return {
        "name": name,
        "bits": int(bits.size),
        "bias": rep.bias,
        "shannon": rep.shannon,
        "min_entropy": rep.min_entropy,
        "mcv_800_90b": mcv,
        "nist_passed": sum(r.passed for r in battery),
        "nist_total": len(battery),
        "battery": battery,
    }


if go:
    with st.spinner("Sampling the quantum circuit..."):
        if apply_noise:
            nm = noise.thermal_noise_model()
            q_matrix = noise.sample_with_noise(n_qubits, n_shots, nm, seed=int(seed))
            label = "Quantum (noisy sim)"
        else:
            q_matrix = core.sample_bitmatrix(n_qubits, n_shots, seed=int(seed))
            label = "Quantum (ideal sim)"
        q_bits = q_matrix.reshape(-1)

    sources = [summarise(label, q_bits)]
    if apply_noise:
        sources.append(summarise(f"{label} + Von Neumann",
                                 mitigation.von_neumann_extract(q_bits)))
    sources.append(summarise("MT19937 (PRNG)",
                             classical.mt19937_bits(q_bits.size, seed=int(seed))))
    sources.append(summarise("secrets (CSPRNG)", classical.secrets_bits(q_bits.size)))

    st.subheader("Quantum source at a glance")
    q = sources[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bias |p₁-0.5|", f"{q['bias']:.4f}")
    c2.metric("Shannon H", f"{q['shannon']:.4f}/bit")
    c3.metric("Min-entropy H∞", f"{q['min_entropy']:.4f}/bit")
    c4.metric("NIST passed", f"{q['nist_passed']}/{q['nist_total']}")

    st.subheader("Source comparison")
    st.dataframe(
        [{"Source": s["name"], "Bits": s["bits"], "Bias": round(s["bias"], 4),
          "Shannon/bit": round(s["shannon"], 5), "H∞/bit": round(s["min_entropy"], 4),
          "800-90B MCV": round(s["mcv_800_90b"], 4),
          "NIST (8)": f"{s['nist_passed']}/{s['nist_total']}"} for s in sources],
        width="stretch", hide_index=True,
    )
    st.caption("NIST column is the 8-test from-scratch subset (fast). The full "
               "15-test battery is too slow to run live — see `results/full_nist.json`.")

    st.subheader("NIST SP 800-22 detail (quantum source)")
    st.dataframe(
        [{"Test": r.name, "p-value": round(r.p_value, 4),
          "Result": "PASS" if r.passed else "FAIL"} for r in q["battery"]],
        width="stretch", hide_index=True,
    )

    st.subheader("Bit distribution (first 4096 bits)")
    st.bar_chart({"0s vs 1s": [int((q_bits[:4096] == 0).sum()),
                               int((q_bits[:4096] == 1).sum())]})

    st.subheader("Application: quantum-seeded AES-256-GCM")
    msg = st.text_input("Message to encrypt", "Sealed with a quantum-random key.")
    if msg:
        key, sealed = crypto_demo.encrypt(q_bits, msg.encode(), associated_data=b"qrng-demo")
        pt = crypto_demo.decrypt(key, sealed, associated_data=b"qrng-demo")
        st.write("Quantum bits → SHA-256 (conditioning) → 256-bit key → AES-256-GCM:")
        st.code(f"key (hex)  = {key.hex()}\n"
                f"nonce      = {sealed.nonce.hex()}\n"
                f"ciphertext = {sealed.ciphertext.hex()}\n"
                f"decrypted  = {pt.decode()!r}  ({'round-trip OK' if pt == msg.encode() else 'FAIL'})")

    with st.expander("Why this beats a PRNG even when both pass NIST"):
        import random
        rng = random.Random(0xC0FFEE)
        observed = [rng.getrandbits(32) for _ in range(624)]
        predicted = classical.mt19937_next_bit_attack(observed)
        actual = rng.getrandbits(32)
        st.write(
            "A good PRNG (MT19937) passes the whole NIST battery and scores ~1.0 "
            "min-entropy — output statistics cannot tell it from true randomness. "
            "The real separation is **predictability given internal state**:"
        )
        st.code(
            f"From 624 consecutive MT19937 outputs, predicted next output = {predicted}\n"
            f"Actual next output                                = {actual}\n"
            f"-> {'EXACT MATCH — PRNG fully predictable' if predicted == actual else 'mismatch'}",
        )
        st.write("The quantum source has no internal state to recover.")
else:
    st.info("Set parameters in the sidebar and click **Generate quantum bits**.")
