"""Run the corrected full 15-test SP 800-22 battery on the quantum source and one
classical baseline at >= 1,000,000 bits, caching results to results/full_nist.json.

Slow (Berlekamp-Massey on 1M bits is minutes); intended to be run once and the
JSON reused by the README / Streamlit app rather than recomputed live.

    python scripts/full_nist_run.py            # 1M bits, ideal simulator
    python scripts/full_nist_run.py --noise    # 1M bits, hardware-like noise
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qrng import classical, core, nist_full, noise  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--shots", type=int, default=125_000, help="8 x 125000 = 1,000,000 bits")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.noise:
        nm = noise.thermal_noise_model()
        q_bits = noise.sample_with_noise(args.qubits, args.shots, nm, seed=args.seed).reshape(-1)
        q_label = "Quantum (noisy simulator)"
    else:
        q_bits = core.generate_quantum_bits(args.qubits, args.shots, seed=args.seed)
        q_label = "Quantum (ideal simulator)"

    sources = [(q_label, q_bits),
               ("secrets (CSPRNG)", classical.secrets_bits(q_bits.size))]

    out = {}
    for label, bits in sources:
        print(f"\n=== {label}: full battery on {bits.size:,} bits ===", flush=True)
        t0 = time.time()
        rep = nist_full.run_full_battery(bits)
        print(f"  {rep.headline}  ({time.time()-t0:.0f}s)")
        for nm_, p, ok in rep.results:
            print(f"      {'PASS' if ok else 'FAIL'}  {nm_:34s} p={p:.4f}")
        if rep.skipped:
            print(f"  skipped: {', '.join(rep.skipped)}")
        out[label] = {"n_bits": rep.n_bits, "headline": rep.headline,
                      "eligible": rep.eligible, "passed": rep.passed,
                      "total": rep.total_tests, "results": rep.results,
                      "skipped": rep.skipped}

    (RESULTS / "full_nist.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'full_nist.json'}")


if __name__ == "__main__":
    main()
