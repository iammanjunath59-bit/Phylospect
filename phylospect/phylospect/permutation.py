"""
Parametric bootstrap null for PHYLOSPECT.

Primary test statistic: Nonsynonymous Operator Excess (NOE)
-----------------------------------------------------------
NOE(b) = sum_{(i,j): nonsyn single-nt} [Qb[i,j] - Qpool[i,j]]

Measures the excess nonsynonymous substitution rate on branch b relative
to the pooled background operator.  Positive NOE = elevated nonsynonymous
rates, consistent with episodic positive selection.  Negative NOE =
suppressed rates, consistent with strong purifying selection.

Parametric bootstrap null
--------------------------
1. From the observed alignment, estimate Qpool (the background generator).
2. For each of n_perm bootstrap replicates:
   a. Simulate a neutral codon alignment under Q0 on the same tree
      (all branches neutral — this is the null hypothesis).
   b. Estimate Qb_null for each branch via stochastic mapping.
   c. Compute NOE_null against the FIXED observed Qpool.
3. p-value = fraction of bootstrap NOE_null >= observed NOE.

Design properties
-----------------
- Same estimator (stochastic mapping) for observed and null — no asymmetry.
- Continuous null distribution with n_perm values.
- Same n_perm null replicates serve all branches simultaneously — the
  bootstrap overhead is paid once per alignment call regardless of tree size.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def nonsynonymous_operator_excess(
    Qb: np.ndarray,
    Qpool: np.ndarray,
    nsyn_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Nonsynonymous Operator Excess (NOE) — primary PHYLOSPECT test statistic.

    NOE = sum_{(i,j): nonsyn single-nt} [Qb[i,j] - Qpool[i,j]]

    Positive NOE indicates elevated nonsynonymous substitution rates on the
    branch relative to the pooled background — consistent with episodic
    positive selection.  Negative NOE indicates suppression — consistent
    with strong purifying selection.

    Parameters
    ----------
    Qb       : (61, 61) branch-specific codon rate matrix.
    Qpool    : (61, 61) pooled background codon rate matrix.
    nsyn_mask: (61, 61) bool mask of nonsynonymous single-nucleotide pairs.
               Computed from gy94 if not supplied.

    Returns
    -------
    float
    """
    if nsyn_mask is None:
        from phylospect.gy94 import nonsynonymous_single_nt_mask
        nsyn_mask = nonsynonymous_single_nt_mask()
    return float((Qb - Qpool)[nsyn_mask].sum())


def run_permutation_test(
    Qs: dict[str, np.ndarray],
    Qpool: np.ndarray,
    newick: str,
    Q0: np.ndarray,
    n_sites: int,
    n_perm: int = 50,
    n_sm_samples_null: int = 8,
    focal_branches: Optional[list[str]] = None,
    nsyn_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, dict]:
    """
    Parametric bootstrap permutation test for episodic positive selection.

    Parameters
    ----------
    Qs            : dict {branch_name: Qb (61x61)} from stochastic mapping
                    on the observed alignment.
    Qpool         : (61,61) observed pooled background generator.
    newick        : Newick tree string (same topology as observed).
    Q0            : GY94 neutral rate matrix used for null simulation.
    n_sites       : number of codon sites in the observed alignment.
    n_perm        : number of bootstrap replicates (default 50).
    n_sm_samples_null : stochastic mapping samples per null replicate (default 8).
    focal_branches: branches to test; all branches in Qs if None.
    nsyn_mask     : precomputed (61,61) bool mask; built from gy94 if None.
    rng           : random Generator for reproducibility.

    Returns
    -------
    dict {branch_name: {
        'NOE'       : float   — observed NOE (primary statistic),
        'pvalue'    : float   — fraction of bootstrap NOE >= observed NOE,
        'ICE'       : float   — signed mean of full ΔQb (secondary diagnostic),
        'ICEnorm'   : float   — Frobenius norm of ΔQb (secondary diagnostic),
        'null_stats': ndarray — bootstrap null NOE values (length = n_perm),
    }}

    Notes
    -----
    'ICE' and 'ICEnorm' are secondary diagnostics retained for backward
    compatibility.  The primary test statistic is 'NOE'; use 'pvalue' for
    significance testing.
    """
    from phylospect.gy94 import nonsynonymous_single_nt_mask, uniform_codon_frequencies
    from phylospect.statistics import signed_ICE, ICEnorm
    from phylospect.simulate import parse_newick, simulate_alignment_numpy
    from phylospect.stochastic_mapping import (
        estimate_branch_Qs_stochastic,
        _make_P_cache,
    )

    if rng is None:
        rng = np.random.default_rng()
    if nsyn_mask is None:
        nsyn_mask = nonsynonymous_single_nt_mask()
    if focal_branches is None:
        focal_branches = list(Qs.keys())

    tree    = parse_newick(newick)
    pi_sim  = uniform_codon_frequencies()

    # ── observed statistics ───────────────────────────────────────────────────
    observed = {}
    for b in focal_branches:
        dQ = Qs[b] - Qpool
        observed[b] = {
            'NOE':     float(dQ[nsyn_mask].sum()),
            'ICE':     signed_ICE(dQ),
            'ICEnorm': ICEnorm(dQ),
        }

    # Pre-compute P matrices once — reused for all null replicates
    P_cache_null = _make_P_cache(tree, Q0)

    # ── parametric bootstrap null ─────────────────────────────────────────────
    null_noe = {b: [] for b in focal_branches}

    for _ in range(n_perm):
        rng_rep = np.random.default_rng(rng.integers(10_000_000))

        aln_null = simulate_alignment_numpy(
            tree=tree,
            n_sites=n_sites,
            q_for_branch=lambda node, _Q=Q0: _Q,
            pi_root=pi_sim,
            rng=rng_rep,
        )

        Qs_null, _ = estimate_branch_Qs_stochastic(
            tree, aln_null, Q0,
            n_samples=n_sm_samples_null,
            rng=rng_rep,
            P_cache=P_cache_null,
        )

        for b in focal_branches:
            if b in Qs_null:
                noe_null = float((Qs_null[b] - Qpool)[nsyn_mask].sum())
            else:
                noe_null = 0.0
            null_noe[b].append(noe_null)

    # ── assemble results ──────────────────────────────────────────────────────
    results = {}
    for b in focal_branches:
        null = np.array(null_noe[b])
        noe  = observed[b]['NOE']
        results[b] = {
            'NOE':        noe,
            'pvalue':     float(np.mean(null >= noe)),
            'ICE':        observed[b]['ICE'],
            'ICEnorm':    observed[b]['ICEnorm'],
            'null_stats': null,
        }
    return results


# ── backward-compatibility stubs ─────────────────────────────────────────────

def compute_pvalue(mb_obs: float, null_stats: np.ndarray) -> float:
    """Retained for backward compatibility."""
    return float(np.mean(null_stats >= mb_obs))


def _collect_site_events(node_states, node_names, tree, branch_names):
    """Parsimony-based site event collection (retained for diagnostics)."""
    branch_idx_map = {name: i for i, name in enumerate(branch_names)}
    n_nodes, n_sites = node_states.shape
    name_to_row = {name: i for i, name in enumerate(node_names)}
    site_events  = [[] for _ in range(n_sites)]
    for node in tree.iter_postorder():
        if node.parent is None:
            continue
        bname = node.name
        if bname not in branch_idx_map:
            continue
        bidx = branch_idx_map[bname]
        rc   = name_to_row[bname]
        rp   = name_to_row[node.parent.name]
        for s in range(n_sites):
            cs, ps = int(node_states[rc, s]), int(node_states[rp, s])
            if cs != ps:
                site_events[s].append((ps, cs, bidx))
    return site_events
