"""
Spectral basis construction for PHYLOSPECT.

Two constructors are provided:

  build_spectral_basis_empirical  (PRIMARY — used in all main analyses)
      Learns the basis from ΔQb matrices produced by the same parsimony
      pipeline that PHYLOSPECT applies to real data.  The basis therefore
      lives in the same sparse, discrete, estimation-noise-affected subspace
      as the empirical operator deviations it will be projected onto.

      Pilot experiments showed that a basis built from clean, dense GY94
      perturbations of a theoretical Q₀ was not aligned with empirical ΔQbs,
      because parsimony-estimated Qb matrices are sparse (≈6% filled entries
      at 3000 codon sites) while theoretical perturbations are dense.  Using
      an empirical basis that shares the same sparsity structure eliminates
      this mismatch.

  build_spectral_basis_clean  (SUPPLEMENTARY — robustness check)
      Classic approach: perturb a theoretical neutral GY94 Q₀ analytically,
      compute ΔQ vectors, QR-decompose.  Retained for comparison purposes and
      reported in Supplementary Results S1.1.

Both return an orthonormal basis B of shape (3721, k).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


# ---------------------------------------------------------------------------
# Primary: empirical basis
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
    Learn a spectral basis from empirical ΔQb matrices.

    For each of n_samples replicates:
      1. Simulate a codon alignment under mild selection (ω_perturb on a
         randomly chosen subset of branches, neutral elsewhere) using the
         NumPy simulator.
      2. Run Fitch parsimony to get branch-specific Qb estimates.
      3. Compute ΔQb = Qb − Qpool and flatten to a 3721-dim vector.

    The resulting matrix of difference vectors is QR-decomposed to yield an
    orthonormal basis spanning their dominant directions.

    Parameters
    ----------
    Q0           : (61, 61) neutral GY94 generator — defines kappa, pi, scale
                   (used only as a reference; simulations use Pyvolve/NumPy).
    tree         : parsed TreeNode from phylospect.simulate.parse_newick.
    n_samples    : number of perturbation replicates (default 50).
    n_basis      : number of basis vectors to retain (default 15).
    n_sites      : alignment length for each replicate (default 3000).
    omega_perturb: ω on the perturbed branch during basis learning (default 1.2).
    frac_perturb : fraction of branches randomly selected as foreground
                   for each replicate (default 0.30).
    kappa        : transition/transversion ratio for simulations.
    pi           : codon equilibrium frequencies; if None, uniform.
    rng          : random Generator for reproducibility.

    Returns
    -------
    B : (3721, n_basis) orthonormal basis matrix.
    """
    from phylospect.simulate import (
        simulate_episodic_alignment_numpy,
        _write_labelled_newick,
    )
    from phylospect.parsimony import estimate_branch_Qs, estimate_pooled_Q
    from phylospect.gy94 import uniform_codon_frequencies

    if rng is None:
        rng = np.random.default_rng()
    if pi is None:
        from phylospect.gy94 import uniform_codon_frequencies
        pi = uniform_codon_frequencies()

    # Collect all leaf branch names for random foreground selection.
    leaf_names = [n.name for n in tree.leaves()]

    dim = 61 * 61
    vectors = []

    # Build a plain newick string from the tree (no labels) for re-use.
    from phylospect.simulate import _write_labelled_newick as _wln
    plain_newick = _wln(tree, labelled_nodes=set(), label="fg")

    for s in range(n_samples):
        # Randomly pick ~frac_perturb of leaf branches as foreground.
        n_fg = max(1, int(round(frac_perturb * len(leaf_names))))
        fg = set(rng.choice(leaf_names, size=n_fg, replace=False).tolist())

        # Simulate alignment using foreground_branches parameter directly.
        aln, tree_out = simulate_episodic_alignment_numpy(
            newick=plain_newick,
            n_sites=n_sites,
            omega_background=1.0,
            omega_foreground=omega_perturb,
            foreground_branches=fg,
            kappa=kappa,
            pi=pi,
            rng=rng,
        )

        # Estimate branch Qb matrices via parsimony.
        Qs, _ = estimate_branch_Qs(tree_out, aln, rng=rng)
        Qpool = estimate_pooled_Q(Qs, tree_out)

        # Collect ΔQb vectors for all branches.
        for bname, Qb in Qs.items():
            dQ = Qb - Qpool
            vectors.append(dQ.flatten())

    if len(vectors) < n_basis:
        raise RuntimeError(
            f"Only {len(vectors)} ΔQb vectors collected; need at least {n_basis}. "
            f"Increase n_samples or reduce n_basis."
        )

    X = np.column_stack(vectors)  # shape (3721, len(vectors))
    Q_mat, _ = qr(X, mode="economic")
    return Q_mat[:, :n_basis]


# ---------------------------------------------------------------------------
# Supplementary: clean (analytical) basis — kept for comparison
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

    This is the approach described in early drafts of the manuscript.  Pilot
    experiments (Supplementary S1.1) showed it underperforms the empirical
    basis on parsimony-estimated data because the clean ΔQ vectors are dense
    while empirical ΔQbs are sparse.  Retained as a supplementary robustness
    comparison.

    Parameters
    ----------
    Q0           : (61, 61) neutral GY94 generator.
    n_samples    : number of perturbation replicates.
    n_basis      : number of basis vectors to retain.
    omega_perturb: ω used for perturbations.
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
    nonsyn_ij = np.argwhere(nonsyn_mask)  # (N_nonsyn, 2)
    n_nonsyn = len(nonsyn_ij)
    n_perturb = max(1, int(round(frac_perturb * n_nonsyn)))

    codons = 61
    dim = codons * codons
    vectors = []

    for _ in range(n_samples):
        Qp = Q0.copy()
        # Perturb a random subset of nonsynonymous single-nt entries.
        idx = rng.choice(n_nonsyn, size=n_perturb, replace=False)
        for ii, jj in nonsyn_ij[idx]:
            Qp[ii, jj] *= omega_perturb
        # Recompute diagonal so rows sum to zero.
        for i in range(codons):
            Qp[i, i] = 0.0
            Qp[i, i] = -Qp[i, :].sum()

        delta = (Qp - Q0).flatten()
        vectors.append(delta)

    X = np.column_stack(vectors)  # (3721, n_samples)
    Q_mat, _ = qr(X, mode="economic")
    return Q_mat[:, :n_basis]
