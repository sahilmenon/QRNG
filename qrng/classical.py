"""Classical RNG baselines for comparison against the quantum source.

Two references:
  * MT19937  -- Python's ``random`` module (a non-cryptographic PRNG).
  * secrets  -- the OS CSPRNG (``secrets``), cryptographically secure.

Both emit bits that pass the NIST SP 800-22 battery, which is the whole point:
output-based statistical tests cannot tell a good PRNG from a true RNG. The
distinction lives elsewhere (see ``mt19937_state_recovery`` below and the
min-entropy discussion in the README).
"""

from __future__ import annotations

import random
import secrets

import numpy as np


def mt19937_bits(n_bits: int, seed: int | None = None) -> np.ndarray:
    """Return ``n_bits`` bits (uint8 0/1) from Python's MT19937 PRNG."""
    rng = random.Random(seed)
    # getrandbits is the most direct tap into the Mersenne Twister stream.
    out = np.empty(n_bits, dtype=np.uint8)
    # Pull 32 bits at a time for speed.
    i = 0
    while i < n_bits:
        chunk = rng.getrandbits(32)
        take = min(32, n_bits - i)
        for j in range(take):
            out[i + j] = (chunk >> j) & 1
        i += take
    return out


def secrets_bits(n_bits: int) -> np.ndarray:
    """Return ``n_bits`` bits (uint8 0/1) from the OS CSPRNG (``secrets``)."""
    n_bytes = (n_bits + 7) // 8
    raw = secrets.token_bytes(n_bytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits[:n_bits].copy()


def mt19937_next_bit_attack(observed_outputs: list[int]) -> int:
    """Demonstrate MT19937's defining weakness: full state recovery.

    Given 624 consecutive 32-bit outputs, the Mersenne Twister's internal state
    is fully recoverable and *all* future output is deterministic. This is the
    honest "adversarial randomness" separation between a PRNG and a true RNG --
    output-based statistics (NIST, min-entropy) cannot see it, but it makes
    MT19937 useless for cryptography.

    ``observed_outputs`` must be >= 624 consecutive 32-bit unsigned ints from a
    single MT19937 stream. Returns the predicted next 32-bit output.
    """
    if len(observed_outputs) < 624:
        raise ValueError("need >= 624 consecutive 32-bit outputs to recover state")

    def untemper(y: int) -> int:
        y ^= y >> 18
        y ^= (y << 15) & 0xEFC60000
        # reverse y ^= (y << 7) & 0x9D2C5680
        for _ in range(7):
            y ^= (y << 7) & 0x9D2C5680
        # reverse y ^= y >> 11
        for _ in range(3):
            y ^= y >> 11
        return y & 0xFFFFFFFF

    state = tuple(untemper(x) for x in observed_outputs[-624:]) + (624,)
    clone = random.Random()
    clone.setstate((3, state, None))
    return clone.getrandbits(32)
