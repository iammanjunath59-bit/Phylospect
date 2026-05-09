"""
Fitch parsimony ancestral state reconstruction for PHYLOSPECT.

This module reconstructs ancestral codon states at every internal node
using Fitch parsimony (Fitch 1971), assigns substitutions to specific
branches, and constructs 61×61 empirical rate matrices Qb for each branch.

Role in the pipeline
--------------------
The parsimony estimator is retained for supplementary investigations
(Supplementary Note S1) and as a fast fallback.  The primary pipeline uses
stochastic mapping (stochastic_mapping.py), which produces denser, less
biased Qb estimates by sampling from the full posterior distribution over
substitution histories.

Caveats
-------
Parsimony mapping underestimates substitution counts on long branches
because back-mutations are invisible to parsimony (Felsenstein 1978).
With 61 codon states and typical branch lengths (≤ 0.2 subs/codon), most
Qb entries will be zero or one — the resulting matrices are sparse and noisy.
This is a fundamental limitation of the parsimony estimator; see
stochastic_mapping.py for the recommended alternative.

References
----------
Fitch WM. 1971. Toward defining the course of evolution. Syst Zool. 20:406–416.
Nielsen R. 2002. Mapping mutations on phylogenies. Syst Biol. 51:729–739.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .gy94 import sense_codons, codon_index, STANDARD_CODE
from .simulate import TreeNode


# ── alignment → codon index matrix ───────────────────────────────────────────

def alignment_to_codon_matrix(
    alignment: dict[str, str],
    code: dict[str, str] = STANDARD_CODE,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert {leaf_name: nucleotide_string} to (n_leaves, n_sites) int matrix.

    Returns
    -------
    (matrix, leaf_order)
    """
    leaves = sorted(alignment.keys())
    if not leaves:
        raise ValueError('Empty alignment')
    lengths = {len(seq) for seq in alignment.values()}
    if len(lengths) != 1:
        raise ValueError(f'Leaf sequences have mismatched lengths: {lengths}')
    seq_len = lengths.pop()
    if seq_len % 3 != 0:
        raise ValueError(f'Sequence length {seq_len} is not a multiple of 3')
    n_sites = seq_len // 3

    idx      = codon_index(code)
    n_codons = len(idx)
    mat      = np.empty((len(leaves), n_sites), dtype=np.int16)
    for i, leaf in enumerate(leaves):
        seq = alignment[leaf]
        for s in range(n_sites):
            codon = seq[3*s : 3*s+3]
            if codon not in idx:
                raise ValueError(
                    f'Unrecognised codon {codon!r} at {leaf} site {s} '
                    f'(stop codons or ambiguity not supported)'
                )
            mat[i, s] = idx[codon]
    return mat, leaves


# ── Fitch parsimony ───────────────────────────────────────────────────────────

@dataclass
class FitchResult:
    states: dict[int, int]
    score:  int


def _fitch_site(
    tree:        TreeNode,
    leaf_states: dict[str, int],
    n_codons:    int,
    rng:         np.random.Generator,
) -> FitchResult:
    """
    Fitch two-pass reconstruction for a single codon site.

    Down-pass (post-order): compute per-node codon sets.
      - Leaf: singleton set of the observed codon.
      - Internal: intersection of children if non-empty; union + 1 otherwise.

    Up-pass (pre-order): assign specific codons.
      - Root: any codon in its Fitch set.
      - Non-root: parent's state if in Fitch set (no substitution); otherwise
        any state in the Fitch set.

    Ties broken uniformly at random via rng.
    """
    sets:  dict[int, np.ndarray] = {}
    score: int = 0

    # Down-pass
    for node in tree.iter_postorder():
        if node.is_leaf:
            mask = np.zeros(n_codons, dtype=bool)
            mask[leaf_states[node.name]] = True
            sets[id(node)] = mask
            continue
        child_masks = [sets[id(c)] for c in node.children]
        inter = np.ones(n_codons, dtype=bool)
        for m in child_masks:
            inter &= m
        if inter.any():
            sets[id(node)] = inter
        else:
            union = np.zeros(n_codons, dtype=bool)
            for m in child_masks:
                union |= m
            sets[id(node)] = union
            score += 1

    # Up-pass
    states: dict[int, int] = {}
    root_candidates   = np.flatnonzero(sets[id(tree)])
    states[id(tree)]  = int(rng.choice(root_candidates))

    for node in tree.iter_preorder():
        if node.parent is None:
            continue
        parent_state = states[id(node.parent)]
        node_mask    = sets[id(node)]
        if node_mask[parent_state]:
            states[id(node)] = parent_state
        else:
            states[id(node)] = int(rng.choice(np.flatnonzero(node_mask)))

    return FitchResult(states=states, score=score)


def fitch_reconstruct(
    tree:          TreeNode,
    codon_matrix:  np.ndarray,
    leaf_order:    list[str],
    rng:           np.random.Generator | int | None = None,
    code:          dict[str, str] = STANDARD_CODE,
) -> tuple[np.ndarray, list[str]]:
    """
    Ancestral codon reconstruction at every node for every site.

    Parameters
    ----------
    tree         : rooted phylogenetic tree.
    codon_matrix : (n_leaves, n_sites) int array of leaf codon indices.
    leaf_order   : leaf names in same row order as codon_matrix.
    rng          : random source for tie-breaking (None → default_rng(0)).

    Returns
    -------
    node_states : (n_nodes, n_sites) int array.
    node_names  : list of node names in matching row order.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    n_codons  = len(codon_index(code))
    n_sites   = codon_matrix.shape[1]
    nodes     = list(tree.iter_postorder())
    node_idx  = {id(n): i for i, n in enumerate(nodes)}
    node_names = [n.name for n in nodes]
    leaf_row   = {name: i for i, name in enumerate(leaf_order)}

    node_states = np.empty((len(nodes), n_sites), dtype=np.int16)

    for s in range(n_sites):
        leaf_states = {
            leaf.name: int(codon_matrix[leaf_row[leaf.name], s])
            for leaf in tree.leaves()
        }
        res = _fitch_site(tree, leaf_states, n_codons, rng)
        for n in nodes:
            node_states[node_idx[id(n)], s] = res.states[id(n)]

    return node_states, node_names


# ── substitution counting and Qb estimation ───────────────────────────────────

def count_branch_substitutions(
    tree:        TreeNode,
    node_states: np.ndarray,
    node_names:  list[str],
    code:        dict[str, str] = STANDARD_CODE,
) -> dict[str, np.ndarray]:
    """
    Count i→j substitutions per branch from the ancestral reconstruction.

    Returns
    -------
    dict {branch_name: (61, 61) int count matrix}
    """
    n_codons   = len(codon_index(code))
    node_index = {name: i for i, name in enumerate(node_names)}
    counts: dict[str, np.ndarray] = {}

    for node in tree.iter_postorder():
        if node.parent is None:
            continue
        child_states  = node_states[node_index[node.name]]
        parent_states = node_states[node_index[node.parent.name]]
        diff_mask     = child_states != parent_states
        cm = np.zeros((n_codons, n_codons), dtype=np.int64)
        if diff_mask.any():
            np.add.at(cm, (parent_states[diff_mask], child_states[diff_mask]), 1)
        counts[node.name] = cm

    return counts


def estimate_branch_Q(
    substitution_counts: np.ndarray,
    branch_length:       float,
    n_sites:             int,
    smoothing:           float = 0.0,
) -> np.ndarray:
    """
    Convert a (61, 61) substitution count matrix into an empirical rate
    matrix Qb.

    Qb[i,j] = counts[i,j] / (n_sites * branch_length)   for i ≠ j
    Qb[i,i] = -sum_{j≠i} Qb[i,j]

    Parameters
    ----------
    substitution_counts : (61, 61) integer count matrix.
    branch_length       : expected substitutions per codon along the branch.
    n_sites             : number of codon sites in the alignment.
    smoothing           : Laplace-style additive smoothing for off-diagonals.

    Returns
    -------
    Qb : (61, 61) rate matrix with rows summing to zero.
    """
    counts = substitution_counts.astype(float)
    if smoothing > 0:
        off = ~np.eye(counts.shape[0], dtype=bool)
        counts = counts.copy()
        counts[off] += smoothing

    if branch_length <= 0 or n_sites <= 0:
        return np.zeros_like(counts)

    Q = counts / (float(n_sites) * float(branch_length))
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def estimate_branch_Qs(
    tree:      TreeNode,
    alignment: dict[str, str],
    rng:       np.random.Generator | int | None = None,
    smoothing: float = 0.0,
    code:      dict[str, str] = STANDARD_CODE,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """
    End-to-end: alignment + tree → {branch_name: Qb}.

    Parameters
    ----------
    tree      : rooted phylogenetic tree with branch lengths.
    alignment : {leaf_name: codon_string}.
    rng       : random source for Fitch tie-breaking.
    smoothing : see estimate_branch_Q.

    Returns
    -------
    Qs        : {branch_name: (61, 61) rate matrix}
    sub_totals: {branch_name: total substitutions}
    """
    codon_mat, leaf_order = alignment_to_codon_matrix(alignment, code)
    node_states, node_names = fitch_reconstruct(
        tree, codon_mat, leaf_order, rng=rng, code=code
    )
    sub_counts = count_branch_substitutions(tree, node_states, node_names, code)
    n_sites    = codon_mat.shape[1]
    branch_len = {
        n.name: n.branch_length
        for n in tree.iter_postorder()
        if n.parent is not None
    }

    Qs: dict[str, np.ndarray] = {}
    sub_totals: dict[str, int] = {}
    for branch_name, counts in sub_counts.items():
        bl = branch_len[branch_name]
        Qs[branch_name] = estimate_branch_Q(counts, bl, n_sites, smoothing)
        sub_totals[branch_name] = int(counts.sum())

    return Qs, sub_totals


def estimate_pooled_Q(
    Qs:   dict[str, np.ndarray],
    tree: TreeNode,
) -> np.ndarray:
    """
    Branch-length-weighted average of per-branch rate matrices.

    Qpool = Σ_b (L_b × Q_b) / Σ_b L_b

    Longer branches contribute more weight because they provide more
    substitution events and thus more information about the background rate.
    """
    branches = list(Qs.keys())
    if not branches:
        raise ValueError('No branches supplied')
    branch_len = {
        n.name: n.branch_length
        for n in tree.iter_postorder()
        if n.parent is not None
    }
    acc       = np.zeros_like(next(iter(Qs.values())))
    total_len = 0.0
    for b in branches:
        L = branch_len.get(b, 0.0)
        if L <= 0:
            continue
        acc       += L * Qs[b]
        total_len += L
    if total_len <= 0:
        raise RuntimeError('All branches have non-positive length')
    return acc / total_len
