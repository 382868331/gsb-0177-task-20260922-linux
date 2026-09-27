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


def forward_distance(later: int, earlier: int, k: int) -> int:
    """Generation-aware forward distance between two wrapping FIFO pointers.

    ``later`` and ``earlier`` are ``k+1``-bit binary pointers.  The extra
    high bit is the generation/wrap bit, so distance is measured forward on
    the modulo-``2**(k+1)`` pointer circle rather than by ordinary unsigned
    subtraction or ``min``.  A valid FIFO distance is in ``0..2**k``.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if not 1 <= k <= 5:
        raise ValueError(f"k must be within [1, 5], got {k}")
    for name, value in (("later", later), ("earlier", earlier)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be an int, got {type(value).__name__}"
            )
        if not 0 <= value <= (1 << (k + 1)) - 1:
            raise ValueError(f"{name} must be a {k + 1}-bit pointer")
    return (later - earlier) & ((1 << (k + 1)) - 1)
