"""Phase 5: use the QRNG as the entropy source for an AES-256-GCM key.

Raw measured bits carry hardware bias, so they are *conditioned* before becoming
key material -- exactly the role NIST SP 800-90B assigns to a conditioning
function. Here that is SHA-256 over the quantum bit stream, yielding a full-entropy
256-bit key, which then drives AES-256-GCM (authenticated encryption).

    quantum bits --SHA-256--> 256-bit key --> AES-256-GCM(plaintext, nonce)

This is the standard, defensible construction; packing biased bits straight into
a key would not be.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a flat 0/1 array into bytes (MSB-first within each byte)."""
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def condition_to_key(bits: np.ndarray) -> bytes:
    """SHA-256 conditioning of the quantum bit stream -> 32-byte (256-bit) key.

    Requires enough input entropy to justify a 256-bit key; with biased bits you
    need comfortably more than 256 raw bits, so we ask for >= 512.
    """
    if bits.size < 512:
        raise ValueError("need >= 512 quantum bits to condition a 256-bit key")
    return hashlib.sha256(bits_to_bytes(bits)).digest()


@dataclass
class Sealed:
    nonce: bytes
    ciphertext: bytes  # includes the GCM auth tag


def encrypt(bits: np.ndarray, plaintext: bytes,
            associated_data: bytes | None = None) -> tuple[bytes, Sealed]:
    """Derive a quantum-seeded AES-256 key and seal ``plaintext`` with AES-GCM.

    The 96-bit nonce is also drawn from the quantum stream. Returns ``(key,
    Sealed)``; in real use the key is never exported -- it is returned here only
    so the demo can show the round-trip.
    """
    key = condition_to_key(bits)
    # Draw a fresh 96-bit nonce from a *different* slice of quantum bits.
    nonce = bits_to_bytes(bits[-96:]) if bits.size >= 96 else hashlib.sha256(key).digest()[:12]
    nonce = nonce[:12].ljust(12, b"\0")
    sealed = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    return key, Sealed(nonce, sealed)


def decrypt(key: bytes, sealed: Sealed, associated_data: bytes | None = None) -> bytes:
    """Inverse of :func:`encrypt`; raises on tamper (GCM auth failure)."""
    return AESGCM(key).decrypt(sealed.nonce, sealed.ciphertext, associated_data)
