# QRNG — Quantum Random Number Generator

Generate provably random bits by measuring qubits in equal superposition, then
hold those bits to the same statistical scrutiny as classical RNGs — on a local
simulator now, on real IBM Quantum hardware when a token is supplied.

## Why quantum randomness is different

Classical generators (`random`/MT19937, even the OS CSPRNG) are *deterministic*
algorithms: given the seed and the algorithm, every output is reproducible. A
qubit placed in superposition with a Hadamard gate has **no defined value** until
measured; by the Born rule the outcome is 0 or 1 with probability exactly ½, and
by Bell's theorem no hidden variable predicts it. That is provable randomness,
not merely unpredictable randomness.

```
H|0⟩ = (|0⟩ + |1⟩)/√2   ──measure──▶   0 or 1, each p = 0.5
```

One shot of an `n`-qubit all-Hadamard circuit yields `n` random bits.

## An honest note on what the tests can and cannot show

A well-designed PRNG (MT19937) passes the **entire NIST SP 800-22 battery** and
scores ≈1.0 bit/bit on output **min-entropy** — the same as the quantum source.
Output-based statistics are *necessary but not sufficient*; they cannot see a
seed. The real PRNG/TRNG separation is **predictability given internal state**:
MT19937's full state is recoverable from 624 consecutive outputs, after which all
future output is exactly predictable (`qrng/classical.py::mt19937_next_bit_attack`).
The quantum source has no such state to recover. This project demonstrates *that*
distinction rather than claiming a min-entropy gap that does not exist.

## Project layout

| Path | What it does |
|------|--------------|
| `qrng/core.py` | Build the all-Hadamard circuit; sample on the local simulator; flatten to a bit array. |
| `qrng/hardware.py` | Submit the same circuit to IBM Quantum hardware (needs an API token). |
| `qrng/noise.py` | Noise model + circuit-depth sweep to study decoherence effects. |
| `qrng/classical.py` | MT19937 and `secrets` baselines + the MT19937 state-recovery attack. |
| `qrng/analysis.py` | Bias, Shannon entropy, min-entropy, autocorrelation, cross-qubit χ². |
| `qrng/nist.py` | 8 core NIST SP 800-22 tests implemented from scratch. |
| `qrng/nist_full.py` | Full 15-test SP 800-22 battery via `nistrng`, with two of its bugs fixed (see below). |
| `qrng/entropy_800_90b.py` | NIST SP 800-90B min-entropy estimators (MCV + Markov). |
| `qrng/crypto_demo.py` | Quantum bits → SHA-256 → AES-256-GCM authenticated encryption. |
| `scripts/run_pipeline.py` | End-to-end: generate → analyse → NIST → 800-90B → AES → compare → plot. |
| `scripts/full_nist_run.py` | The slow 1M-bit full battery, cached to `results/full_nist.json`. |
| `tests/` | Unit tests (run against the simulator, no token needed). |

## Two bugs found and fixed in the `nistrng` library

Treating `nistrng` as a black-box "15/15" oracle would have been wrong — it has
two bugs that make *good* random data fail, both verified and worked around in
`qrng/nist_full.py`:

1. **Shared-array mutation.** `run_all_battery` passes one array to every test;
   several mutate it in place (DFT maps `0 → -1`) or overflow on narrow dtypes,
   silently corrupting later tests so uniform random *fails* DFT/Serial/ApEn/Cusum.
   Fix: run each test on its own fresh `int64` copy.
2. **Linear Complexity binning.** It bins the statistic with `int(T + 2.5)`
   (floor) instead of rounding, so the modal value `T=0` lands in the π=0.125 bin
   instead of the correct π=0.5 bin; χ² explodes and p→0 on any good source. The
   Berlekamp-Massey routine and constants are correct, so `linear_complexity_fixed`
   reuses them and re-bins correctly. Same LC array: nistrng p=0.0000 → fixed p≈0.54.

The 8 from-scratch tests in `qrng/nist.py` are cross-validated against `nistrng`
(8/8 pass/fail agreement; parameter-free tests match p-values exactly).

## Setup

Requires **Python 3.13** (qiskit-aer has no 3.14 wheels yet).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (simulator)

```powershell
python scripts/run_pipeline.py                 # 8-test battery + 800-90B + AES demo
python scripts/run_pipeline.py --noise         # hardware-like noise + Von Neumann recovery
python scripts/full_nist_run.py                # full 15-test battery on 1M bits (slow)
```

## Run on real IBM Quantum hardware

1. Create a free account at <https://quantum.cloud.ibm.com>, then copy your
   **API key** and your **instance CRN** from the dashboard.
2. Save credentials once (kept in `~/.qiskit`, never committed). The legacy
   `channel="ibm_quantum"` was removed — use `ibm_quantum_platform`:
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(
       channel="ibm_quantum_platform",
       token="<API_KEY>", instance="<INSTANCE_CRN>", overwrite=True)
   ```
3. `python scripts/run_pipeline.py --hardware`

The backend is chosen at submission time via `least_busy(...)`; backend names and
free-tier limits drift, so nothing is hardcoded.

## Live demo (shareable link)

`app.py` is a Streamlit front-end that runs the whole pipeline on the simulator —
no token or secrets, so it deploys to a free public URL:

```powershell
streamlit run app.py            # local
```

To publish: push this repo to GitHub, then on
[share.streamlit.io](https://share.streamlit.io) pick the repo and `app.py` —
you get a public `https://<app>.streamlit.app` link. `requirements.txt` and
`runtime.txt` (Python 3.13) are already set up for the build. Hugging Face Spaces
(Streamlit SDK) works the same way.

## Status

- [x] Project scaffold, Python 3.13 venv, dependencies
- [x] Core simulator QRNG (bit-extraction path verified against Qiskit 2.x)
- [x] 8 NIST SP 800-22 tests from scratch + full 15-test battery (nistrng, 2 bugs fixed)
- [x] NIST SP 800-90B min-entropy estimators (MCV + Markov)
- [x] Classical baselines + min-entropy + MT19937 state-recovery attack
- [x] Noise characterisation (thermal depth sweep, cross-qubit χ²) + mitigation
- [x] Phase 5: quantum-seeded AES-256-GCM authenticated encryption
- [x] Streamlit demo (`app.py`) for a shareable public link
- [ ] Hardware submission (code ready in `qrng/hardware.py`; pending IBM token)

`python scripts/run_pipeline.py --noise` produces `results/summary.json`,
`results/nist_pass_rates.png`, and `results/decoherence_sweep.png`.
