"""
PHYLOSPECT end-to-end pipeline.

Single entry point that takes a tree + alignment (or simulation parameters)
and returns a results dict with Mb, p-value, ICE, ICEnorm for every branch.

Typical usage (simulated data)
-------------------------------
from phylospect.pipeline import run_phylospect

results = run_phylospect(
    newick        = "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);",
    alignment     = aln_dict,          # {leaf_name: codon_string}
    B             = spectral_basis,    # pre-built (61^2 x k) basis
    n_perm        = 200,
    rng           = np.random.default_rng(42),
)
# results["A"] = {"Mb": ..., "pvalue": ..., "ICE": ..., "ICEnorm": ..., ...}

Or build + use the basis in one call:

results, B = run_phylospect_with_basis_learning(
    newick        = newick,
    alignment     = aln,
    Q0            = Q0,                # neutral GY94 generator
    basis_type    = "empirical",       # or "clean"
    n_basis       = 15,
    n_perm        = 200,
    rng           = rng,
)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def run_phylospect(
    newick: str,
    alignment: dict[str, str],
    n_perm: int = 50,
    estimator: str = "stochastic",
    n_sm_samples: int = 30,
    n_sm_samples_null: int = 10,
    Q0: Optional[np.ndarray] = None,
    focal_branches: Optional[list[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, dict]:
    """
    Run the full PHYLOSPECT pipeline on a given alignment.

    Steps
    -----
    1. Estimate branch-specific Qb matrices via stochastic mapping.
    2. Compute observed NOE, ICE, ICEnorm for each branch.
    3. Run parametric bootstrap permutation null (n_perm replicates).
    4. Return p-values and effect sizes.

    Parameters
    ----------
    newick             : Newick string.
    alignment          : dict {leaf_name: codon_string}.
    n_perm             : bootstrap replicates for null (default 50).
    estimator          : "stochastic" (recommended) or "parsimony".
    n_sm_samples       : stochastic mapping samples for observed Qb.
    n_sm_samples_null  : stochastic mapping samples per null replicate.
    Q0                 : neutral GY94 generator; built with defaults if None.
    focal_branches     : branches to test; all branches if None.
    rng                : random Generator for reproducibility.

    Returns
    -------
    results : dict {branch_name: {"NOE","pvalue","ICE","ICEnorm","null_stats"}}
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
        Q0 = build_gy94_generator(omega=1.0, kappa=2.0,
                                  pi=uniform_codon_frequencies(), scale=True)

    tree = parse_newick(newick)
    n_sites = len(next(iter(alignment.values()))) // 3

    # --- Qb estimation ---
    if estimator == "stochastic":
        Qs, _ = estimate_branch_Qs_stochastic(
            tree, alignment, Q0, n_samples=n_sm_samples, rng=rng
        )
        Qpool = estimate_pooled_Q_stochastic(Qs, tree)
    else:
        Qs, _ = estimate_branch_Qs(tree, alignment, rng=rng)
        Qpool = estimate_pooled_Q(Qs, tree)

    # --- Parametric bootstrap permutation test ---
    results = run_permutation_test(
        Qs=Qs,
        Qpool=Qpool,
        newick=newick,
        Q0=Q0,
        n_sites=n_sites,
        n_perm=n_perm,
        n_sm_samples_null=n_sm_samples_null,
        focal_branches=focal_branches,
        rng=rng,
    )

    return results
    """
    Run the full PHYLOSPECT pipeline on a given alignment.

    Steps
    -----
    1. Parse tree, run ancestral reconstruction (stochastic mapping or parsimony).
    2. Estimate branch-specific Qb matrices and pooled Qpool.
    3. Compute observed Mb, ICE, ICEnorm for each branch.
    4. Run within-site permutation null (n_perm replicates).
    5. Return p-values and effect sizes for all branches.

    Parameters
    ----------
    newick      : Newick string (plain, no branch labels).
    alignment   : dict {leaf_name: codon_string}.
    B           : (3721, k) orthonormal spectral basis.
    n_perm      : number of permutations for the null distribution.
    estimator   : "stochastic" (default, recommended) or "parsimony".
                  Stochastic mapping produces denser Qb estimates and
                  better-calibrated p-values.
    n_sm_samples: number of posterior samples for stochastic mapping
                  (ignored if estimator="parsimony").
    Q0          : (61, 61) neutral GY94 generator required for stochastic
                  mapping.  If None, built with default parameters.
    rng         : random Generator for reproducibility.

    Returns
    -------
    results : dict {branch_name: {"Mb", "pvalue", "ICE", "ICEnorm",
                                   "null_stats", "deltaQ"}}
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

    tree = parse_newick(newick)

    # --- Qb estimation ---
    if estimator == "stochastic":
        if Q0 is None:
            from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
            Q0 = build_gy94_generator(omega=1.0, kappa=2.0,
                                      pi=uniform_codon_frequencies(), scale=True)
        Qs, _ = estimate_branch_Qs_stochastic(
            tree, alignment, Q0, n_samples=n_sm_samples, rng=rng
        )
        Qpool = estimate_pooled_Q_stochastic(Qs, tree)
    else:
        Qs, _ = estimate_branch_Qs(tree, alignment, rng=rng)
        Qpool  = estimate_pooled_Q(Qs, tree)

    # --- Permutation test (Qs-matrix relabeling, no parsimony needed) ---
    results = run_permutation_test(
        Qs=Qs,
        Qpool=Qpool,
        n_perm=n_perm,
        rng=rng,
    )

    return results


def run_phylospect_with_basis_learning(
    newick: str,
    alignment: dict[str, str],
    Q0: np.ndarray,
    basis_type: str = "empirical",
    n_basis: int = 15,
    n_basis_samples: int = 50,
    n_perm: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> tuple[dict[str, dict], np.ndarray]:
    """
    Convenience wrapper: build the spectral basis then run the pipeline.

    Parameters
    ----------
    Q0           : (61, 61) neutral GY94 generator.
    basis_type   : "empirical" (recommended) or "clean" (supplementary).
    n_basis      : number of basis vectors to retain (default 15).
    n_basis_samples : number of perturbation replicates for basis learning.

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

    tree = parse_newick(newick)
    n_sites = len(next(iter(alignment.values()))) // 3

    if basis_type == "empirical":
        B = build_spectral_basis_empirical(
            Q0=Q0, tree=tree,
            n_samples=n_basis_samples, n_basis=n_basis,
            n_sites=min(n_sites, 2000),   # cap basis-learning sites for speed
            rng=rng,
        )
    elif basis_type == "clean":
        B = build_spectral_basis_clean(
            Q0=Q0, n_samples=n_basis_samples, n_basis=n_basis, rng=rng
        )
    else:
        raise ValueError(f"Unknown basis_type '{basis_type}'. Use 'empirical' or 'clean'.")

    results = run_phylospect(
        newick=newick, alignment=alignment,
        B=B, n_perm=n_perm, rng=rng,
    )
    return results, B


def build_shared_basis(
    Q0: np.ndarray,
    newick: str,
    basis_type: str = "empirical",
    n_basis: int = 15,
    n_basis_samples: int = 50,
    n_sites_for_basis: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build a single spectral basis to be reused across many replicates.

    For experiments with many replicates, build the basis once here and
    pass it to run_phylospect() in each replicate, rather than rebuilding
    it every time.

    Parameters
    ----------
    Q0                : neutral GY94 generator.
    newick            : tree topology (used only for empirical basis).
    basis_type        : "empirical" or "clean".
    n_basis           : number of basis vectors.
    n_basis_samples   : perturbation replicates for basis learning.
    n_sites_for_basis : alignment length used during basis learning.
    rng               : random Generator.

    Returns
    -------
    B : (3721, n_basis) orthonormal basis.
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
