"""
Spectral basis construction for PHYLOSPECT (Supplementary Note S1).

This module supports the spectral projection investigation documented in
Supplementary Note S1.  It is NOT part of the primary NOE-based analysis;
the functions here are called only by the Mb-related supplementary code
(pipeline.run_phylospect_with_basis_learning and experiment 07_basis_sensitivity.py).

Two constructors are provided:

  build_spectral_basis_empirical
      Learns the basis from ΔQb matrices produced by the stochastic mapping
      pipeline, using an independent held-out set of neutral replicates.
      The basis lives in the same subspace as the empirical operator
      deviations it is projected onto.

  build_spectral_basis_clean
      Classic approach: perturb a theoretical neutral GY94 Q₀ analytically,
      compute ΔQ vectors, QR-decompose.  Retained for comparison purposes.
      See Supplementary Note S1 for calibration results with both approaches.

Both return an orthonormal basis B of shape (3721, k).
The calibration failure of Mb under both basis constructions is documented
in Supplementary Notes S1 and S2.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


# ---------------------------------------------------------------------------
# Empirical basis (Supplementary Note S1)
# ---------------------------------------------------------------------------

def build_spectral_basis_empirical(
    Q0: np.ndarray,
    tree,
    n_samples: int = 50,
    n_basis: int = 15,
    n_sites: int = 3000,
    omega_perturb: float = 1.2,
    frac_perturb: float = 0.30,
    kappa: float = 2.0,
    pi: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build a spectral basis from empirical ΔQb matrices (Supplementary Note S1).

    For each of n_samples replicates:
      1. Simulate a codon alignment under mild selection (omega_perturb on a
         randomly chosen subset of branches, neutral elsewhere).
      2. Run stochastic mapping to estimate branch-specific Qb matrices.
      3. Compute ΔQb = Qb − Qpool and flatten to a 3721-dim vector.

    The resulting matrix of vectors is QR-decomposed to yield an orthonormal
    basis spanning their dominant directions.

    For calibration properties of Mb under this basis, see Supplementary
    Note S1 (circular-basis setup) and Supplementary Note S2 (independent
    held-out basis, Experiment 07).

    Parameters
    ----------
    Q0           : (61, 61) neutral GY94 generator.
    tree         : parsed TreeNode from phylospect.simulate.parse_newick.
    n_samples    : number of perturbation replicates for basis learning.
    n_basis      : number of basis vectors to retain (default 15).
    n_sites      : alignment length for each replicate.
    omega_perturb: ω on perturbed branches during basis learning.
    frac_perturb : fraction of leaf branches randomly selected as foreground.
    kappa        : transition/transversion ratio for simulations.
    pi           : codon equilibrium frequencies; uniform if None.
    rng          : random Generator for reproducibility.

    Returns
    -------
    B : (3721, n_basis) orthonormal basis matrix.
    """
    from phylospect.simulate import simulate_episodic_alignment_numpy, _write_labelled_newick
    from phylospect.parsimony import estimate_branch_Qs, estimate_pooled_Q
    from phylospect.gy94 import uniform_codon_frequencies

    if rng is None:
        rng = np.random.default_rng()
    if pi is None:
        pi = uniform_codon_frequencies()

    leaf_names   = [n.name for n in tree.leaves()]
    plain_newick = _write_labelled_newick(tree, labelled_nodes=set(), label="fg")
    vectors      = []

    for _ in range(n_samples):
        n_fg = max(1, int(round(frac_perturb * len(leaf_names))))
        fg   = set(rng.choice(leaf_names, size=n_fg, replace=False).tolist())

        aln, tree_out = simulate_episodic_alignment_numpy(
            newick=plain_newick, n_sites=n_sites,
            omega_background=1.0, omega_foreground=omega_perturb,
            foreground_branches=fg, kappa=kappa, pi=pi, rng=rng,
        )
        Qs, _  = estimate_branch_Qs(tree_out, aln, rng=rng)
        Qpool  = estimate_pooled_Q(Qs, tree_out)

        for bname, Qb in Qs.items():
            vectors.append((Qb - Qpool).flatten())

    if len(vectors) < n_basis:
        raise RuntimeError(
            f"Only {len(vectors)} ΔQb vectors collected; need at least {n_basis}. "
            f"Increase n_samples or reduce n_basis."
        )

    X = np.column_stack(vectors)
    Q_mat, _ = qr(X, mode="economic")
    return Q_mat[:, :n_basis]


# ---------------------------------------------------------------------------
# Clean (analytical) basis (Supplementary Note S1 — comparison)
# ---------------------------------------------------------------------------

def build_spectral_basis_clean(
    Q0: np.ndarray,
    n_samples: int = 50,
    n_basis: int = 15,
    omega_perturb: float = 1.2,
    frac_perturb: float = 0.30,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build a spectral basis from analytical GY94 perturbations of Q₀.

    Retained for comparison with the empirical basis; see Supplementary
    Note S1 for calibration results.

    Parameters
    ----------
    Q0           : (61, 61) neutral GY94 generator.
    n_samples    : number of perturbation replicates.
    n_basis      : number of basis vectors to retain.
    omega_perturb: ω scaling applied to perturbed entries.
    frac_perturb : fraction of nonsynonymous entries perturbed per sample.
    rng          : random Generator.

    Returns
    -------
    B : (3721, n_basis) orthonormal basis matrix.
    """
    from phylospect.gy94 import nonsynonymous_single_nt_mask

    if rng is None:
        rng = np.random.default_rng()

    nonsyn_mask = nonsynonymous_single_nt_mask()
    nonsyn_ij   = np.argwhere(nonsyn_mask)
    n_nonsyn    = len(nonsyn_ij)
    n_perturb   = max(1, int(round(frac_perturb * n_nonsyn)))
    codons      = 61
    vectors     = []

    for _ in range(n_samples):
        Qp  = Q0.copy()
        idx = rng.choice(n_nonsyn, size=n_perturb, replace=False)
        for ii, jj in nonsyn_ij[idx]:
            Qp[ii, jj] *= omega_perturb
        for i in range(codons):
            Qp[i, i] = 0.0
            Qp[i, i] = -Qp[i, :].sum()
        vectors.append((Qp - Q0).flatten())

    X = np.column_stack(vectors)
    Q_mat, _ = qr(X, mode="economic")
    return Q_mat[:, :n_basis]
