"""Gray code helpers for (k+1)-bit FIFO pointers.

Pointers are k+1 bits: k bits index a depth-2**k memory word plus one extra
high bit that records wrap count.  Adjacent pointer values differ in one Gray
bit, so a multi-bit synchronizer can never sample an illegal intermediate code
(other than a momentarily old, but always legal, code).
"""


def binary_to_gray(value: int) -> int:
    """Convert a non-negative binary integer to its reflected Gray code.

    The caller fixes the width via the input range; this function imposes no
    masking.  Negative values raise ``ValueError``.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"binary_to_gray expects int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"binary_to_gray expects value >= 0, got {value}")
    return value ^ (value >> 1)


def gray_to_binary(gray: int) -> int:
    """Inverse of :func:`binary_to_gray` for non-negative codes."""
    if not isinstance(gray, int) or isinstance(gray, bool):
        raise TypeError(
            f"gray_to_binary expects int, got {type(gray).__name__}"
        )
    if gray < 0:
        raise ValueError(f"gray_to_binary expects gray >= 0, got {gray}")
    bits = gray
    while bits:
        bits >>= 1
        gray ^= bits
    return gray


def forward_distance(leader: int, follower: int, modulus: int) -> int:
    """Wrap-aware forward distance from follower to leader using generations."""
    if (
        isinstance(leader, bool)
        or isinstance(follower, bool)
        or isinstance(modulus, bool)
        or not all(isinstance(v, int) for v in (leader, follower, modulus))
    ):
        raise TypeError("forward_distance expects integer arguments")
    if leader < 0 or follower < 0 or modulus <= 0:
        raise ValueError("forward_distance expects valid pointer arguments")
    return (leader - follower) % modulus
