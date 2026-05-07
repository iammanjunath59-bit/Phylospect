"""
Branch-specific summary statistics for PHYLOSPECT.

Three statistics are computed per branch from the operator difference
ΔQb = Qb - Qpool:

  ICE  (Interventional Codon Effect, signed)
       Mean of all entries in ΔQb.  Positive = overall rate elevation,
       negative = overall rate reduction, near-zero = balanced or symmetric
       restructuring.  May cancel under symmetric ω inflation; use ICEnorm
       for magnitude-based inference.

  ICEnorm  (Frobenius magnitude)
       ||ΔQb||_F = sqrt(sum of squared entries).  Insensitive to sign
       cancellation; robust indicator of total magnitude of operator deviation.

  Mb   (branchwise spectral projection statistic)
       ||B^T vec(ΔQb)||_2 where B is the learned spectral basis.  Projects
       the 3721-dim deviation vector onto biologically plausible directions;
       primary hypothesis-testing statistic in PHYLOSPECT.

Note on naming
--------------
Earlier drafts called ICE the "Interventional Causal Effect" and drew on
Pearl / Hernán-Robins causal inference machinery.  Both peer reviewers flagged
the word "causal" as inappropriate (no DAG, no identification assumption).
The name has been changed to "Interventional Codon Effect"; the formulas are
unchanged.  Citations to Pearl 2009 / Hernán & Robins 2020 have been removed.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def compute_deltaQ(Qb: np.ndarray, Qpool: np.ndarray) -> np.ndarray:
    """
    Compute the operator difference ΔQb = Qb − Qpool.

    Parameters
    ----------
    Qb    : (61, 61) branch-specific codon generator.
    Qpool : (61, 61) pooled background codon generator.

    Returns
    -------
    deltaQ : (61, 61) ndarray.
    """
    return Qb - Qpool


def signed_ICE(deltaQ: np.ndarray) -> float:
    """
    Signed Interventional Codon Effect (ICE).

    Defined as the mean of all entries in ΔQb:
        ICE = (1 / 61²) Σ_{i,j} ΔQ_{ij}

    Positive values indicate net rate elevation; negative values indicate net
    reduction.  May be near zero even under strong selection if rate increases
    and decreases cancel (symmetric restructuring).  This is not evidence
    against selection — use ICEnorm in that case.

    Parameters
    ----------
    deltaQ : (61, 61) operator difference matrix.

    Returns
    -------
    float
    """
    return float(deltaQ.mean())


def ICEnorm(deltaQ: np.ndarray) -> float:
    """
    Frobenius-norm magnitude of the operator deviation (ICEnorm).

    ICEnorm = ||ΔQb||_F = sqrt(Σ_{i,j} ΔQ_{ij}²)

    Insensitive to sign cancellation: all squared deviations contribute
    positively.  Robust indicator of total magnitude of perturbation even
    when directional effects are balanced.

    Parameters
    ----------
    deltaQ : (61, 61) operator difference matrix.

    Returns
    -------
    float
    """
    return float(np.linalg.norm(deltaQ, ord="fro"))


def projection_stat(deltaQ: np.ndarray, B: np.ndarray) -> float:
    """
    Branchwise spectral projection statistic Mb.

    Mb = ||B^T vec(ΔQb)||_2

    where vec(ΔQb) is the 3721-dim vectorisation of ΔQb (row-major) and B
    is the (3721, k) orthonormal spectral basis.  Measures how strongly
    ΔQb aligns with biologically plausible directions of operator deviation.

    Parameters
    ----------
    deltaQ : (61, 61) operator difference matrix.
    B      : (3721, k) orthonormal spectral basis.

    Returns
    -------
    float  — Euclidean norm of the projected vector.
    """
    vec = deltaQ.flatten()
    projected = B.T @ vec
    return float(np.linalg.norm(projected))


def compute_all_stats(
    Qb: np.ndarray,
    Qpool: np.ndarray,
    B: np.ndarray,
) -> dict[str, float]:
    """
    Convenience wrapper: compute ΔQb and all three statistics in one call.

    Returns
    -------
    dict with keys 'ICE', 'ICEnorm', 'Mb', and the deltaQ matrix.
    """
    dQ = compute_deltaQ(Qb, Qpool)
    return {
        "deltaQ": dQ,
        "ICE": signed_ICE(dQ),
        "ICEnorm": ICEnorm(dQ),
        "Mb": projection_stat(dQ, B),
    }
