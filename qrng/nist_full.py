"""Full NIST SP 800-22 battery (all 15 tests) built on the ``nistrng`` library,
with two of its bugs worked around.

``nistrng`` is convenient but NOT a trustworthy oracle out of the box -- two
independent bugs were found and verified here:

  1. ``run_all_battery`` shares one array across every test, and several tests
     mutate it in place (DFT maps 0 -> -1) or overflow on narrow dtypes, silently
     corrupting later tests so that *uniform random data fails* DFT/Serial/ApEn/
     Cusum. Worked around by running each test on its own fresh int64 copy.

  2. The **Linear Complexity** test bins its statistic with ``int(T + 2.5)``
     (floor) instead of rounding, so the modal value T=0 lands in the pi=0.125
     bin instead of the correct pi=0.5 bin. chi-square then explodes and p -> 0
     on *any* good source. Its Berlekamp-Massey routine and constants are correct,
     so we reuse the former and re-bin correctly in :func:`linear_complexity_fixed`.

Honesty rules enforced here:
  * A length-ineligible test is *skipped*, not failed -- we report
    "<passed>/<eligible> eligible" plus how many of 15 were eligible, never a
    bare "15/15".
  * Random Excursions can return N/A even when eligible; reported as skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaincc


def linear_complexity_fixed(bits: np.ndarray, significance: float = 0.01) -> tuple[float, bool]:
    """Linear Complexity test with the binning bug fixed.

    Reuses nistrng's verified Berlekamp-Massey, but bins the statistic
    T_i = (-1)^M (L_i - mu) + 2/9 by *rounding* to the nearest NIST category
    (clamped to [0, K]) instead of flooring. For the default M=512 every T_i is
    an integer (mu + 2/9 cancels), so rounding is exact; for odd M the half-
    integer boundary convention would need care.
    """
    from nistrng import SP800_22R1A_BATTERY

    t = SP800_22R1A_BATTERY["linear_complexity"]
    m = t._pattern_length
    mu = t._mu
    k = t._freedom_degrees
    pi = np.asarray(t._probabilities, dtype=np.float64)

    seq = np.asarray(bits, dtype=np.int64)
    n_blocks = seq.size // m
    lc = np.fromiter(
        (t._berlekamp_massey(seq[i * m:(i + 1) * m]) for i in range(n_blocks)),
        dtype=np.int64, count=n_blocks,
    )
    tickets = ((-1.0) ** m) * (lc - mu) + (2.0 / 9.0)
    # correct NIST binning: category = round(T) + 3, clamped to [0, K]
    idx = np.clip(np.rint(tickets).astype(int) + 3, 0, k)
    freqs = np.bincount(idx, minlength=k + 1)[: k + 1]
    chi2 = float(np.sum((freqs - n_blocks * pi) ** 2 / (n_blocks * pi)))
    p = float(gammaincc(k / 2.0, chi2 / 2.0))
    return p, (p >= significance)


@dataclass
class FullBatteryReport:
    n_bits: int
    total_tests: int                       # always 15
    eligible: int                          # how many ran
    passed: int                            # how many of the eligible passed
    results: list[tuple[str, float, bool]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        return (f"{self.passed}/{self.eligible} eligible NIST SP 800-22 tests passed "
                f"({self.eligible}/{self.total_tests} eligible at {self.n_bits:,} bits)")


def run_full_battery(bits: np.ndarray, significance: float = 0.01) -> FullBatteryReport:
    """Run all eligible SP 800-22 tests on fresh copies; return a structured report."""
    from nistrng import SP800_22R1A_BATTERY, check_eligibility_all_battery

    seq = np.asarray(bits, dtype=np.int64)
    eligible = check_eligibility_all_battery(seq, SP800_22R1A_BATTERY)
    results: list[tuple[str, float, bool]] = []
    skipped: list[str] = [name for name in SP800_22R1A_BATTERY if name not in eligible]
    passed = 0

    for name, test in eligible.items():
        if name == "linear_complexity":
            # use our bug-fixed implementation, not nistrng's broken binning
            score, ok = linear_complexity_fixed(seq, significance)
            disp_name = "Linear Complexity (fixed)"
        else:
            test.significance_value = significance
            # fresh copy per test -> avoids nistrng's shared-array mutation/overflow bug
            result, _ = test.run(seq.copy())
            score = float(result.score)
            ok = bool(result.passed)
            disp_name = result.name
        if not np.isfinite(score):
            skipped.append(disp_name)         # e.g. Random Excursions returned N/A
            continue
        results.append((disp_name, round(score, 4), ok))
        passed += int(ok)

    return FullBatteryReport(
        n_bits=int(seq.size),
        total_tests=len(SP800_22R1A_BATTERY),
        eligible=len(results),
        passed=passed,
        results=results,
        skipped=skipped,
    )
