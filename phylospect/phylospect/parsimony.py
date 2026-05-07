"""
Empirical estimation of branch-specific codon substitution generators.

Given a simulated or observed codon alignment and a phylogenetic tree, this
module reconstructs ancestral codon states at every internal node (Fitch
parsimony), assigns substitutions to specific branches, and constructs a
61x61 empirical rate matrix Qb for each branch in the tree.

Scientific context
------------------
The original PHYLOSPECT manuscript (Methods 2.3) claims to use "maximum
parsimony mapping of substitutions onto the tree" to build branch-specific
generators, but the original code skipped this entirely — Qb was simulated
directly from a random matrix with no reference to an alignment or tree.
This module is the real implementation.

Caveats — read before using
---------------------------
1. Parsimony mapping UNDERESTIMATES substitution counts on long branches
   (Felsenstein 1978; back-mutations are invisible to parsimony). This bias
   is real and acknowledged in the manuscript's Discussion §4.7.

2. With 61 codon states and typical branch lengths (<= 0.2 subs/codon),
   most entries of any given Qb will be zero or one. The resulting matrices
   are very sparse and noisy. PHYLOSPECT compensates for this by
   aggregating across codon sites and projecting onto a spectral basis,
   but the underlying sparsity is a fundamental limit.

3. Tie-breaking at ambiguous Fitch resolutions is random by default. Running
   the same alignment twice gives slightly different Qbs. This is usually
   fine (the downstream spectral statistic is robust) but callers who need
   deterministic results should pass ``rng`` explicitly.

References
----------
Fitch, W.M., 1971. Toward defining the course of evolution. Syst. Zool.
  20, 406-416.
Nielsen, R., 2002. Mapping mutations on phylogenies. Syst. Biol. 51, 729-739.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .gy94 import sense_codons, codon_index, STANDARD_CODE
from .simulate import TreeNode


# ---------------------------------------------------------------------------
# Alignment -> per-site codon index matrix
# ---------------------------------------------------------------------------

def alignment_to_codon_matrix(
    alignment: dict[str, str],
    code: dict[str, str] = STANDARD_CODE,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert a dict {leaf_name: nucleotide_string} into a (n_leaves, n_sites)
    matrix of codon indices (int). Returns (matrix, leaf_order).
    """
    leaves = sorted(alignment.keys())
    if not leaves:
        raise ValueError("Empty alignment")
    lengths = {len(seq) for seq in alignment.values()}
    if len(lengths) != 1:
        raise ValueError(f"Leaf sequences have mismatched lengths: {lengths}")
    seq_len = lengths.pop()
    if seq_len % 3 != 0:
        raise ValueError(f"Sequence length {seq_len} is not a multiple of 3")
    n_sites = seq_len // 3

    idx = codon_index(code)
    n_codons = len(idx)
    mat = np.empty((len(leaves), n_sites), dtype=np.int16)
    for i, leaf in enumerate(leaves):
        seq = alignment[leaf]
        for s in range(n_sites):
            codon = seq[3 * s:3 * s + 3]
            if codon not in idx:
                raise ValueError(
                    f"Unrecognised codon {codon!r} at {leaf} site {s} "
                    f"(stop codons or ambiguity not supported)"
                )
            mat[i, s] = idx[codon]
    assert np.all(mat < n_codons)
    return mat, leaves


# ---------------------------------------------------------------------------
# Fitch parsimony: ancestral state reconstruction
# ---------------------------------------------------------------------------

@dataclass
class FitchResult:
    """Result of Fitch reconstruction for one codon site across a tree."""
    # Mapping from node id(node) -> inferred codon index at that node.
    states: dict[int, int]
    # Total parsimony score (minimum number of substitutions) for this site.
    score: int


def _fitch_site(
    tree: TreeNode,
    leaf_states: dict[str, int],
    n_codons: int,
    rng: np.random.Generator,
) -> FitchResult:
    """
    Fitch two-pass reconstruction for a single codon site.

    Step 1 (down-pass, post-order): compute per-node codon SETS that
    minimize substitutions in the subtree below.
      - Leaf: singleton set containing the observed codon.
      - Internal: if children's sets intersect, take the intersection;
        otherwise take the union and add 1 to the parsimony score.

    Step 2 (up-pass, pre-order): pick a specific codon for each node.
      - Root: pick any codon in its Fitch set.
      - Non-root: if parent's chosen state is in node's Fitch set, use it
        (avoids a substitution on this branch); otherwise pick any state
        in the node's Fitch set.

    Ties are broken uniformly at random. Rng must be supplied.
    """
    # Fitch sets: we represent each set as a boolean mask of length n_codons.
    # This is memory-hungry (61 bits per node per site) but trivially fast
    # with NumPy.
    sets: dict[int, np.ndarray] = {}
    score = 0

    # Down-pass.
    for node in tree.iter_postorder():
        if node.is_leaf:
            s = leaf_states[node.name]
            mask = np.zeros(n_codons, dtype=bool)
            mask[s] = True
            sets[id(node)] = mask
            continue

        child_masks = [sets[id(c)] for c in node.children]
        # Intersection of all children.
        inter = np.ones(n_codons, dtype=bool)
        for m in child_masks:
            inter &= m
        if inter.any():
            sets[id(node)] = inter
        else:
            # Union, plus one substitution.
            union = np.zeros(n_codons, dtype=bool)
            for m in child_masks:
                union |= m
            sets[id(node)] = union
            score += 1

    # Up-pass: pick specific codon per node.
    states: dict[int, int] = {}
    # Root: pick uniformly from its Fitch set.
    root_mask = sets[id(tree)]
    root_candidates = np.flatnonzero(root_mask)
    states[id(tree)] = int(rng.choice(root_candidates))

    # Pre-order traversal: parents resolved before children.
    for node in tree.iter_preorder():
        if node.parent is None:
            continue
        parent_state = states[id(node.parent)]
        node_mask = sets[id(node)]
        if node_mask[parent_state]:
            # Use parent's state (no substitution on this branch).
            states[id(node)] = parent_state
        else:
            # Pick uniformly from node's Fitch set.
            candidates = np.flatnonzero(node_mask)
            states[id(node)] = int(rng.choice(candidates))

    return FitchResult(states=states, score=score)


def fitch_reconstruct(
    tree: TreeNode,
    codon_matrix: np.ndarray,
    leaf_order: list[str],
    rng: np.random.Generator | int | None = None,
    code: dict[str, str] = STANDARD_CODE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct ancestral codon states at every node for every codon site.

    Parameters
    ----------
    tree : TreeNode
        Rooted phylogenetic tree.
    codon_matrix : ndarray of shape (n_leaves, n_sites), dtype int
        Observed codon indices at leaves.
    leaf_order : list[str]
        Names of leaves, in the same order as rows of codon_matrix.
    rng : optional
        Random source for tie-breaking. If None, uses default_rng(0).

    Returns
    -------
    node_states : ndarray of shape (n_nodes, n_sites), dtype int
        Inferred codon index at each node for each site.
    node_names : list of length n_nodes
        Names of nodes in the same order as rows of node_states.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    n_codons = len(codon_index(code))
    n_sites = codon_matrix.shape[1]

    # Fix a canonical node ordering (post-order).
    nodes = list(tree.iter_postorder())
    node_index = {id(n): i for i, n in enumerate(nodes)}
    node_names = [n.name for n in nodes]

    # Leaf lookup: leaf name -> row in codon_matrix.
    leaf_row = {name: i for i, name in enumerate(leaf_order)}

    node_states = np.empty((len(nodes), n_sites), dtype=np.int16)

    # This loop is the expensive part; ~ n_sites * n_nodes boolean ops.
    # For 500 sites * ~8 nodes it's fast enough that we don't vectorize further.
    for s in range(n_sites):
        # Build leaf_states for this site.
        leaf_states = {}
        for leaf in tree.leaves():
            leaf_states[leaf.name] = int(codon_matrix[leaf_row[leaf.name], s])
        res = _fitch_site(tree, leaf_states, n_codons, rng)
        for n in nodes:
            node_states[node_index[id(n)], s] = res.states[id(n)]

    return node_states, node_names


# ---------------------------------------------------------------------------
# Substitution assignment and Qb estimation
# ---------------------------------------------------------------------------

def count_branch_substitutions(
    tree: TreeNode,
    node_states: np.ndarray,
    node_names: list[str],
    code: dict[str, str] = STANDARD_CODE,
) -> dict[str, np.ndarray]:
    """
    For each non-root branch, count the number of i->j substitutions inferred
    from the ancestral reconstruction across all sites.

    Parameters
    ----------
    tree : TreeNode
    node_states : ndarray of shape (n_nodes, n_sites)
        Result of ``fitch_reconstruct``.
    node_names : list of length n_nodes

    Returns
    -------
    counts : dict
        Maps "branch name" (the descendant node's name) to a (61, 61) int
        count matrix, where counts[i, j] is the number of inferred i->j
        substitutions on that branch.
    """
    n_codons = len(codon_index(code))
    node_index = {name: i for i, name in enumerate(node_names)}

    counts: dict[str, np.ndarray] = {}
    for node in tree.iter_postorder():
        if node.parent is None:
            continue
        child_row = node_index[node.name]
        parent_row = node_index[node.parent.name]
        child_states = node_states[child_row]
        parent_states = node_states[parent_row]
        # Count i->j only where they differ.
        diff_mask = child_states != parent_states
        if not diff_mask.any():
            counts[node.name] = np.zeros((n_codons, n_codons), dtype=np.int64)
            continue
        i_vals = parent_states[diff_mask]
        j_vals = child_states[diff_mask]
        cm = np.zeros((n_codons, n_codons), dtype=np.int64)
        np.add.at(cm, (i_vals, j_vals), 1)
        counts[node.name] = cm
    return counts


def estimate_branch_Q(
    substitution_counts: np.ndarray,
    branch_length: float,
    n_sites: int,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    Convert a (61, 61) count matrix of inferred substitutions on a branch
    into an empirical rate matrix Qb.

    Rate estimator
    --------------
      Qb[i, j] = counts[i, j] / (n_sites * branch_length)   (i != j)
      Qb[i, i] = -sum_j Qb[i, j]

    This is the standard Poisson-process MLE for a rate, treating the
    observed i->j substitution count as Poisson(rate * exposure), where
    exposure = n_sites * branch_length. We take the expected-time-at-state
    approximation: all n_sites codons are exposed for the full branch
    duration, regardless of which codon they spend most of their time in.
    This is a known simplification but is standard and matches how
    empirical Q matrices are typically reported in the literature.

    Parameters
    ----------
    substitution_counts : ndarray of shape (61, 61)
        Integer counts of i->j inferred substitutions.
    branch_length : float
        Expected substitutions per codon along the branch (from the tree).
    n_sites : int
        Number of codon sites in the alignment.
    smoothing : float
        Laplace-style smoothing added to off-diagonal entries before
        normalization, in units of counts. Default 0 (no smoothing).
        Pass a small positive value (e.g. 0.01) to avoid exact zeros when
        downstream code needs strictly positive rates.

    Returns
    -------
    Qb : ndarray of shape (61, 61)
        Empirical rate matrix. Rows sum to zero.
    """
    counts = substitution_counts.astype(float)
    if smoothing > 0:
        # Smooth off-diagonals only.
        off = ~np.eye(counts.shape[0], dtype=bool)
        counts = counts.copy()
        counts[off] += smoothing

    if branch_length <= 0 or n_sites <= 0:
        # Degenerate branch (zero-length or empty alignment): return zeros.
        return np.zeros_like(counts)

    exposure = float(n_sites) * float(branch_length)
    Q = counts / exposure
    # Zero out any diagonal set by smoothing, then set so rows sum to zero.
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def estimate_branch_Qs(
    tree: TreeNode,
    alignment: dict[str, str],
    rng: np.random.Generator | int | None = None,
    smoothing: float = 0.0,
    code: dict[str, str] = STANDARD_CODE,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """
    End-to-end: alignment + tree -> {branch_name: Qb}.

    Parameters
    ----------
    tree : TreeNode
        Rooted phylogenetic tree. Must have branch lengths.
    alignment : dict[str, str]
        Leaf codon sequences.
    rng : random source for Fitch tie-breaking.
    smoothing : see ``estimate_branch_Q``.

    Returns
    -------
    Qs : dict mapping branch name -> (61, 61) rate matrix
    sub_totals : dict mapping branch name -> total substitutions on that branch
        (useful for diagnostics and reporting).
    """
    codon_mat, leaf_order = alignment_to_codon_matrix(alignment, code)
    node_states, node_names = fitch_reconstruct(
        tree, codon_mat, leaf_order, rng=rng, code=code
    )
    sub_counts = count_branch_substitutions(tree, node_states, node_names, code)

    n_sites = codon_mat.shape[1]
    # Map branch name -> branch length.
    branch_len = {
        n.name: n.branch_length
        for n in tree.iter_postorder()
        if n.parent is not None
    }

    Qs: dict[str, np.ndarray] = {}
    sub_totals: dict[str, int] = {}
    for branch_name, counts in sub_counts.items():
        bl = branch_len[branch_name]
        Qs[branch_name] = estimate_branch_Q(
            counts, bl, n_sites, smoothing=smoothing
        )
        sub_totals[branch_name] = int(counts.sum())
    return Qs, sub_totals


def estimate_pooled_Q(
    Qs: dict[str, np.ndarray],
    tree: TreeNode,
) -> np.ndarray:
    """
    Pool per-branch rate matrices into a single background Qpool by
    averaging weighted by branch length.

    Formally: Qpool = sum_b (L_b * Q_b) / sum_b L_b, where L_b is branch
    length. This is the MLE if the branches shared a common rate matrix:
    longer branches contribute more information and thus more weight.

    Rationale for length-weighting
    ------------------------------
    Equivalent reparameterization: since Q_b = (count_b / L_b) / n_sites
    for each branch, L_b * Q_b = count_b / n_sites. So the length-weighted
    average is the pooled count matrix divided by the total pooled exposure,
    which is exactly the MLE under the shared-rate assumption.
    """
    branches = list(Qs.keys())
    if not branches:
        raise ValueError("No branches supplied")
    total_len = 0.0
    acc = np.zeros_like(next(iter(Qs.values())))
    branch_len = {
        n.name: n.branch_length
        for n in tree.iter_postorder()
        if n.parent is not None
    }
    for b in branches:
        L = branch_len[b]
        if L <= 0:
            continue
        acc += L * Qs[b]
        total_len += L
    if total_len <= 0:
        raise RuntimeError("All branches have nonpositive length")
    return acc / total_len
