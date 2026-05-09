"""
Branch-specific summary statistics for PHYLOSPECT.

Primary statistic
-----------------
NOE  (Nonsynonymous Operator Excess)
     Computed in permutation.py directly on the nonsynonymous single-
     nucleotide codon subspace.  NOE is the primary test statistic used
     in all main-text experiments and empirical analyses.

Secondary / diagnostic statistics (computed per branch from ΔQb = Qb - Qpool)
--------------
signed_ICE
     Mean of ALL entries in ΔQb (not restricted to the nonsynonymous
     subspace).  Retained as a signed diagnostic summary; positive = net
     rate elevation, negative = net rate reduction.  Note: this is NOT
     the same as NOE — NOE is restricted to nonsynonymous single-nucleotide
     pairs in permutation.py, whereas signed_ICE averages all 61×61 entries.

ICEnorm
     ||ΔQb||_F — Frobenius norm magnitude of the full operator deviation.
     Unsigned; insensitive to sign cancellation.

Mb  (spectral projection statistic)
     ||B^T vec(ΔQb)||_2 where B is the learned spectral basis.
     Investigated in Supplementary Note S1; not used as a primary test
     statistic due to calibration difficulties documented there.
"""

from __future__ import annotations
import numpy as np


def compute_deltaQ(Qb: np.ndarray, Qpool: np.ndarray) -> np.ndarray:
    """
    Operator deviation ΔQb = Qb − Qpool.

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
    Signed mean of all entries in ΔQb (secondary diagnostic).

    Defined as the mean over ALL 61×61 entries of ΔQb:
        signed_ICE = mean(ΔQb)

    This is a whole-matrix signed summary and is NOT the same as NOE,
    which is restricted to the nonsynonymous single-nucleotide codon pairs
    (computed in permutation.py).  Use this as a diagnostic; use NOE from
    the permutation results dict for hypothesis testing.

    Positive values indicate net rate elevation; negative values indicate
    net rate reduction across the full operator.

    Parameters
    ----------
    deltaQ : (61, 61) operator difference matrix.

    Returns
    -------
    float
    """
    return float(deltaQ.mean())


# Public alias so callers can import NOE directly from this module.
# The actual subspace-restricted NOE used for hypothesis testing is
# computed inside permutation.run_permutation_test().
NOE = signed_ICE


def ICEnorm(deltaQ: np.ndarray) -> float:
    """
    Frobenius-norm magnitude of the operator deviation.

    ICEnorm = ||ΔQb||_F = sqrt(Σ_{i,j} ΔQ_{ij}²)

    Unsigned indicator of total operator perturbation magnitude.
    Insensitive to sign cancellation.

    Parameters
    ----------
    deltaQ : (61, 61) operator difference matrix.

    Returns
    -------
    float
    """
    return float(np.linalg.norm(deltaQ, ord='fro'))


def projection_stat(deltaQ: np.ndarray, B: np.ndarray) -> float:
    """
    Spectral projection statistic Mb (Supplementary Note S1).

    Mb = ||B^T vec(ΔQb)||_2

    where vec(ΔQb) is the 3721-dim row-major vectorisation of ΔQb and B
    is the (3721, k) orthonormal spectral basis.  Documented in
    Supplementary Note S1; not used in main-text analyses.

    Parameters
    ----------
    deltaQ : (61, 61) operator difference matrix.
    B      : (3721, k) orthonormal spectral basis.

    Returns
    -------
    float
    """
    return float(np.linalg.norm(B.T @ deltaQ.flatten()))


def compute_all_stats(
    Qb: np.ndarray,
    Qpool: np.ndarray,
    B: np.ndarray,
) -> dict:
    """
    Compute ΔQb and all secondary statistics in one call.

    Returns dict with keys 'deltaQ', 'ICE', 'ICEnorm', 'Mb'.
    For the primary NOE test statistic, see permutation.run_permutation_test().
    """
    dQ = compute_deltaQ(Qb, Qpool)
    return {
        'deltaQ':  dQ,
        'ICE':     signed_ICE(dQ),
        'ICEnorm': ICEnorm(dQ),
        'Mb':      projection_stat(dQ, B),
    }
