"""
07_basis_sensitivity.py
=======================
Experiment 07 — Spectral basis dimension sensitivity (Supplementary Note S2).

Tests whether the calibration and power properties of the spectral projection
statistic Mb depend on the number of retained singular directions k.

For each k in {5, 10, 15, 25}:
  - 15 neutral replicates  (omega=1.0 everywhere)  → FPR at alpha=0.05
  - 15 selection replicates (omega=3.0 on branch A) → power at alpha=0.05

Outputs:
  results/07_basis_sensitivity.csv

Usage (from repo root):
    python experiments/07_basis_sensitivity.py

Seed: 700 (fixed for reproducibility)
"""

from __future__ import annotations
import sys, pathlib, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phylospect.gy94          import build_gy94_generator, uniform_codon_frequencies
from phylospect.simulate      import simulate_episodic_alignment_numpy, parse_newick
from phylospect.stochastic_mapping import estimate_branch_Qs_stochastic, estimate_pooled_Q_stochastic
try:
    from phylospect.statistics import projection_stat
except (ImportError, AttributeError):
    def projection_stat(dQ: np.ndarray, basis: np.ndarray) -> float:
        """Inline fallback: M_b = ||B.T @ vec(ΔQ_b)||_2"""
        return float(np.linalg.norm(basis.T @ dQ.flatten()))

# ── inline basis builder — does not depend on phylospect.spectral ─────────────
def build_spectral_basis(delta_Qs: list, k: int) -> np.ndarray:
    """
    Build an orthonormal basis retaining the top-k singular directions from
    a list of operator deviation matrices (ΔQ_b).

    Each ΔQ_b is a (61, 61) numpy array.  The function stacks them as columns
    of a data matrix, runs SVD, and returns the first k left singular vectors
    as a (3721, k) basis matrix B such that M_b = ||B.T @ vec(ΔQ_b)||_2.

    This is identical to what phylospect.spectral.build_spectral_basis does
    internally, written inline so the experiment does not depend on that module
    having the function name defined.
    """
    vecs = np.array([dq.flatten() for dq in delta_Qs]).T  # shape (3721, n_samples)
    U, _, _ = np.linalg.svd(vecs, full_matrices=False)
    return U[:, :k]   # (3721, k)

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ── parameters ────────────────────────────────────────────────────────────────
NEWICK           = "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);"
N_SITES          = 1000
KAPPA            = 2.0
N_SM             = 20          # stochastic mapping samples per observed alignment
N_SM_NULL        = 8           # stochastic mapping samples per bootstrap replicate
N_PERM           = 50          # parametric bootstrap replicates
N_REPS_NEUTRAL   = 15          # neutral replicates per k
N_REPS_SELECTION = 15          # selection replicates per k
OMEGA_BG         = 1.0
OMEGA_FG         = 3.0         # selection on branch A
FOREGROUND       = {'A'}
ALPHA            = 0.05
K_VALUES         = [5, 10, 15, 25]
BASIS_N_NEUTRAL  = 30          # neutral replicates used to build spectral basis (held-out)
SEED             = 700

pi = uniform_codon_frequencies()
Q0 = build_gy94_generator(omega=1.0, kappa=KAPPA, pi=pi, scale=True)

# ── helper: compute Mb p-value for one alignment ──────────────────────────────
def mb_pvalue(aln: dict[str, str], tree, basis: np.ndarray,
              rng: np.random.Generator) -> float:
    """
    Estimate Mb on branch A for the observed alignment, then compare against
    n_perm parametric bootstrap replicates.  Returns (pvalue, mb_obs, mb_null_mean).
    """
    # Observed Mb on branch A
    Qbs_obs, _ = estimate_branch_Qs_stochastic(
        tree, aln, Q0, pi=pi, n_samples=N_SM, rng=rng)
    Qpool_obs = estimate_pooled_Q_stochastic(Qbs_obs, tree)
    dQ_obs = Qbs_obs.get('A', Qbs_obs[list(Qbs_obs)[0]]) - Qpool_obs
    mb_obs = projection_stat(dQ_obs, basis)

    # Bootstrap null
    mb_null = []
    newick = NEWICK   # use global
    for _ in range(N_PERM):
        null_aln, _ = simulate_episodic_alignment_numpy(
            newick=newick, n_sites=N_SITES,
            omega_background=1.0, omega_foreground=1.0,
            foreground_branches=None,
            kappa=KAPPA, pi=pi, rng=rng)
        null_tree = parse_newick(newick)
        Qbs_null, _ = estimate_branch_Qs_stochastic(
            null_tree, null_aln, Q0, pi=pi, n_samples=N_SM_NULL, rng=rng)
        Qpool_null = estimate_pooled_Q_stochastic(Qbs_null, null_tree)
        dQ_null = Qbs_null.get('A', Qbs_null[list(Qbs_null)[0]]) - Qpool_null
        mb_null.append(projection_stat(dQ_null, basis))

    mb_null = np.array(mb_null)
    pvalue = np.mean(mb_null >= mb_obs)
    return float(pvalue), float(mb_obs), float(mb_null.mean())


def main():
    rng_master = np.random.default_rng(SEED)
    rows = []

    print(f"Experiment 07 — Basis dimension sensitivity")
    print(f"  k values:       {K_VALUES}")
    print(f"  Neutral reps:   {N_REPS_NEUTRAL} per k")
    print(f"  Selection reps: {N_REPS_SELECTION} per k")
    print(f"  n_sites:        {N_SITES}")
    print(f"  n_perm:         {N_PERM}")
    print(f"  seed:           {SEED}")
    print()

    for k in K_VALUES:
        print(f"── k = {k} ──────────────────────────────────")
        t0 = time.time()

        # ── Build spectral basis from held-out neutral replicates ──────────────
        print(f"  Building basis from {BASIS_N_NEUTRAL} held-out neutral replicates...")
        delta_Qs = []
        for i in range(BASIS_N_NEUTRAL):
            aln, _ = simulate_episodic_alignment_numpy(
                newick=NEWICK, n_sites=N_SITES,
                omega_background=1.0, omega_foreground=1.0,
                foreground_branches=None,
                kappa=KAPPA, pi=pi, rng=rng_master)
            tree_tmp = parse_newick(NEWICK)
            Qbs, _ = estimate_branch_Qs_stochastic(
                tree_tmp, aln, Q0, pi=pi, n_samples=N_SM_NULL, rng=rng_master)
            Qpool_tmp = estimate_pooled_Q_stochastic(Qbs, tree_tmp)
            for br, Qb in Qbs.items():
                delta_Qs.append(Qb - Qpool_tmp)
        basis = build_spectral_basis(delta_Qs, k=k)
        print(f"  Basis shape: {basis.shape}")

        # ── Neutral replicates ─────────────────────────────────────────────────
        neutral_pvals  = []
        neutral_obs    = []
        neutral_null   = []
        for rep in range(N_REPS_NEUTRAL):
            aln, _ = simulate_episodic_alignment_numpy(
                newick=NEWICK, n_sites=N_SITES,
                omega_background=1.0, omega_foreground=1.0,
                foreground_branches=None,
                kappa=KAPPA, pi=pi, rng=rng_master)
            tree_rep = parse_newick(NEWICK)
            pv, mb_obs_val, mb_null_mean = mb_pvalue(aln, tree_rep, basis, rng_master)
            neutral_pvals.append(pv)
            neutral_obs.append(mb_obs_val)
            neutral_null.append(mb_null_mean)
            rows.append({
                'k': k, 'condition': 'neutral', 'rep': rep + 1,
                'pvalue': pv, 'mb_obs': mb_obs_val, 'mb_null_mean': mb_null_mean,
                'significant': int(pv <= ALPHA)
            })

        fpr = np.mean(np.array(neutral_pvals) <= ALPHA)
        ratio = np.mean(neutral_obs) / np.mean(neutral_null)
        print(f"  Neutral  — FPR={fpr:.3f}  mean p={np.mean(neutral_pvals):.3f}"
              f"  obs/null ratio={ratio:.3f}")

        # ── Selection replicates ───────────────────────────────────────────────
        select_pvals = []
        select_obs   = []
        select_null  = []
        for rep in range(N_REPS_SELECTION):
            aln, _ = simulate_episodic_alignment_numpy(
                newick=NEWICK, n_sites=N_SITES,
                omega_background=OMEGA_BG, omega_foreground=OMEGA_FG,
                foreground_branches=FOREGROUND,
                kappa=KAPPA, pi=pi, rng=rng_master)
            tree_rep = parse_newick(NEWICK)
            pv, mb_obs_val, mb_null_mean = mb_pvalue(aln, tree_rep, basis, rng_master)
            select_pvals.append(pv)
            select_obs.append(mb_obs_val)
            select_null.append(mb_null_mean)
            rows.append({
                'k': k, 'condition': 'selection', 'rep': rep + 1,
                'pvalue': pv, 'mb_obs': mb_obs_val, 'mb_null_mean': mb_null_mean,
                'significant': int(pv <= ALPHA)
            })

        power = np.mean(np.array(select_pvals) <= ALPHA)
        ratio_s = np.mean(select_obs) / np.mean(select_null)
        print(f"  Selection — Power={power:.3f}  mean p={np.mean(select_pvals):.3f}"
              f"  obs/null ratio={ratio_s:.3f}")
        print(f"  Time: {time.time() - t0:.1f}s")
        print()

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / '07_basis_sensitivity.csv'
    df.to_csv(out, index=False)
    print(f"Results saved to {out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'k':>4}  {'FPR':>6}  {'Power':>6}  {'obs/null (neutral)':>18}  {'obs/null (select)':>18}")
    for k in K_VALUES:
        sub_n = df[(df.k == k) & (df.condition == 'neutral')]
        sub_s = df[(df.k == k) & (df.condition == 'selection')]
        fpr   = sub_n.significant.mean()
        power = sub_s.significant.mean()
        rn    = sub_n.mb_obs.mean() / sub_n.mb_null_mean.mean()
        rs    = sub_s.mb_obs.mean() / sub_s.mb_null_mean.mean()
        print(f"{k:>4}  {fpr:>6.3f}  {power:>6.3f}  {rn:>18.3f}  {rs:>18.3f}")


if __name__ == '__main__':
    main()
