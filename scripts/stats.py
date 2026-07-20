"""Pre-registered statistics helpers (WP10). Wilson score intervals as
specified in PREREGISTRATION_DRAFT.md §5; no other CI method may be used
for the primary metrics."""
import math


def wilson(k: int, n: int, z: float = 1.959963984540054):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def fmt_ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}]"
