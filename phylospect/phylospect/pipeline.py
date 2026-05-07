"""
phylospect/pipeline.py
======================
Public API for PHYLOSPECT.

Single entry point: run_phylospect() takes a Newick tree string and a codon
alignment and returns a per-branch dictionary of NOE values and p-values.

Typical usage
-------------
    import numpy as np
    from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
    from phylospect.pipeline import run_phylospect

    pi  = uniform_codon_frequencies()
    Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
    aln = {'A': 'ATGCAG...', 'B': 'ATGCAA...', ...}   # codon strings

    results = run_phylospect(
        newick    = '((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);',
        alignment = aln,
        Q0        = Q0,
        n_perm    = 100,
        rng       = np.random.default_rng(42),
    )

    for branch, r in sorted(results.items()):
        sig = ' *' if r['pvalue'] <= 0.05 else ''
        print(f"{branch:<12} NOE={r['NOE']:+.4f}  p={r['pvalue']:.3f}{sig}")

The spectral projection functions (run_phylospect_with_basis_learning,
build_shared_basis) are retained for the supplementary Mb investigation
documented in Supplementary Note S1.  They are not part of the primary
analysis and are not called by any main-text experiment script.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ── primary public API ────────────────────────────────────────────────────────

def run_phylospect(
    newick:            str,
    alignment:         dict[str, str],
    n_perm:            int                           = 50,
    estimator:         str                           = "stochastic",
    n_sm_samples:      int                           = 20,
    n_sm_samples_null: int                           = 8,
    Q0:                Optional[np.ndarray]          = None,
    focal_branches:    Optional[list[str]]           = None,
    rng:               Optional[np.random.Generator] = None,
) -> dict[str, dict]:
    """
    Run the full PHYLOSPECT pipeline on a codon alignment.

    Steps
    -----
    1. Parse the Newick tree.
    2. Estimate branch-specific codon rate matrices Q_b via stochastic
       mapping (Nielsen 2002; Bollback 2006).
    3. Compute the pooled background operator Q_pool as a branch-length-
       weighted average of all Q_b.
    4. For each branch, compute the observed Nonsynonymous Operator Excess
       (NOE): the signed mean of (Q_b - Q_pool) restricted to nonsynonymous
       single-nucleotide codon pairs.
    5. Run n_perm parametric bootstrap replicates: simulate neutral GY94
       alignments (omega = 1.0), re-estimate Q_b_null, compute NOE_null
       against the fixed observed Q_pool.
    6. P-value = fraction of bootstrap NOE_null values >= observed NOE.

    Parameters
    ----------
    newick : str
        Newick-format tree string with branch lengths.  Leaf names must
        match the keys of ``alignment``.
    alignment : dict[str, str]
        Dictionary mapping taxon name to codon sequence string.  Length
        must be a multiple of 3; no stop codons.
    n_perm : int, default 50
        Number of parametric bootstrap replicates for the null NOE
        distribution.  Use 50 for genome-scale screening; 500–1,000 for
        confirmatory single-gene analysis.
    estimator : str, default "stochastic"
        Branch generator estimation method.  "stochastic" (recommended)
        uses stochastic mapping; "parsimony" uses Fitch parsimony and is
        retained for supplementary investigations only.
    n_sm_samples : int, default 20
        Stochastic mapping samples per observed alignment.
    n_sm_samples_null : int, default 8
        Stochastic mapping samples per bootstrap null replicate.  Lower
        than n_sm_samples by design: per-replicate noise is diluted across
        the bootstrap ensemble.
    Q0 : np.ndarray or None, default None
        Background GY94 rate matrix (61 x 61).  If None, built with
        omega=1.0, kappa=2.0, and uniform codon frequencies.
    focal_branches : list[str] or None, default None
        Subset of branch names to test.  If None, all non-root branches
        are tested.
    rng : np.random.Generator or None
        Random number generator.  If None, a new default Generator is
        created (results will not be reproducible across runs).

    Returns
    -------
    dict[str, dict]
        One entry per tested branch, keyed by branch name.  Each value
        contains:

        ``NOE`` : float
            Signed mean operator deviation on the branch across all
            nonsynonymous single-nucleotide codon pairs.  Positive =
            elevated nonsynonymous rates (consistent with positive
            selection); negative = suppression (consistent with strong
            purifying selection).
        ``pvalue`` : float
            Proportion of bootstrap null NOE values >= observed NOE.
        ``ICE`` : float
            Alias for NOE (retained for backward compatibility).
        ``ICEnorm`` : float
            L2 norm of the full operator deviation vector, restricted to
            the nonsynonymous subspace (unsigned magnitude summary).
        ``null_stats`` : list[float]
            Bootstrap null NOE values (length = n_perm).

    Notes
    -----
    The same n_perm null alignments serve as the null for every branch
    simultaneously; the bootstrap overhead is paid once per alignment call
    regardless of tree size.

    For empirical datasets where background omega differs substantially
    from 1.0, calibration should be verified on neutral permutations of
    the data before interpreting results.
    """
    from phylospect.simulate import parse_newick
    from phylospect.parsimony import estimate_branch_Qs, estimate_pooled_Q
    from phylospect.stochastic_mapping import (
        estimate_branch_Qs_stochastic,
        estimate_pooled_Q_stochastic,
    )
    from phylospect.permutation import run_permutation_test

    if rng is None:
        rng = np.random.default_rng()
    if Q0 is None:
        from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
        Q0 = build_gy94_generator(
            omega=1.0, kappa=2.0,
            pi=uniform_codon_frequencies(), scale=True
        )

    tree = parse_newick(newick)
    n_sites = len(next(iter(alignment.values()))) // 3

    # ── estimate branch generators ────────────────────────────────────────────
    if estimator == "stochastic":
        Qs, _ = estimate_branch_Qs_stochastic(
            tree, alignment, Q0, n_samples=n_sm_samples, rng=rng
        )
        Qpool = estimate_pooled_Q_stochastic(Qs, tree)
    else:
        Qs, _  = estimate_branch_Qs(tree, alignment, rng=rng)
        Qpool  = estimate_pooled_Q(Qs, tree)

    # ── parametric bootstrap NOE test ─────────────────────────────────────────
    results = run_permutation_test(
        Qs                = Qs,
        Qpool             = Qpool,
        newick            = newick,
        Q0                = Q0,
        n_sites           = n_sites,
        n_perm            = n_perm,
        n_sm_samples_null = n_sm_samples_null,
        focal_branches    = focal_branches,
        rng               = rng,
    )

    return results


# ── supplementary spectral projection functions ───────────────────────────────
# The functions below support the spectral projection investigation documented
# in Supplementary Note S1.  They are not called by any main-text experiment
# and are not part of the primary NOE-based analysis.

def run_phylospect_with_basis_learning(
    newick:          str,
    alignment:       dict[str, str],
    Q0:              np.ndarray,
    basis_type:      str                           = "empirical",
    n_basis:         int                           = 15,
    n_basis_samples: int                           = 50,
    n_perm:          int                           = 200,
    rng:             Optional[np.random.Generator] = None,
) -> tuple[dict[str, dict], np.ndarray]:
    """
    [Supplementary] Build a spectral basis then run the Mb pipeline.

    Used exclusively for the spectral projection investigation described in
    Supplementary Note S1.  For the primary NOE analysis, use run_phylospect().

    Parameters
    ----------
    Q0           : (61, 61) neutral GY94 generator.
    basis_type   : "empirical" (recommended) or "clean".
    n_basis      : number of basis vectors to retain (default 15).
    n_basis_samples : perturbation replicates for basis learning.

    Returns
    -------
    (results_dict, B)
    """
    from phylospect.simulate import parse_newick
    from phylospect.spectral import (
        build_spectral_basis_empirical,
        build_spectral_basis_clean,
    )

    if rng is None:
        rng = np.random.default_rng()

    tree    = parse_newick(newick)
    n_sites = len(next(iter(alignment.values()))) // 3

    if basis_type == "empirical":
        B = build_spectral_basis_empirical(
            Q0=Q0, tree=tree,
            n_samples=n_basis_samples, n_basis=n_basis,
            n_sites=min(n_sites, 2000),
            rng=rng,
        )
    elif basis_type == "clean":
        B = build_spectral_basis_clean(
            Q0=Q0, n_samples=n_basis_samples, n_basis=n_basis, rng=rng
        )
    else:
        raise ValueError(
            f"Unknown basis_type '{basis_type}'. Use 'empirical' or 'clean'."
        )

    results = run_phylospect(
        newick=newick, alignment=alignment,
        Q0=Q0, n_perm=n_perm, rng=rng,
    )
    return results, B


def build_shared_basis(
    Q0:                 np.ndarray,
    newick:             str,
    basis_type:         str                           = "empirical",
    n_basis:            int                           = 15,
    n_basis_samples:    int                           = 50,
    n_sites_for_basis:  int                           = 2000,
    rng:                Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    [Supplementary] Build a spectral basis to reuse across many replicates.

    Used exclusively for the spectral projection investigation described in
    Supplementary Note S1.  Build the basis once and pass it to repeated
    calls rather than rebuilding each time.

    Returns
    -------
    B : (3721, n_basis) orthonormal basis matrix.
    """
    from phylospect.simulate import parse_newick
    from phylospect.spectral import (
        build_spectral_basis_empirical,
        build_spectral_basis_clean,
    )

    if rng is None:
        rng = np.random.default_rng()

    if basis_type == "empirical":
        tree = parse_newick(newick)
        return build_spectral_basis_empirical(
            Q0=Q0, tree=tree,
            n_samples=n_basis_samples, n_basis=n_basis,
            n_sites=n_sites_for_basis,
            rng=rng,
        )
    else:
        return build_spectral_basis_clean(
            Q0=Q0, n_samples=n_basis_samples, n_basis=n_basis, rng=rng
        )
