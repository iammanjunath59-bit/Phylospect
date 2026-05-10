"""
Codon alignment simulation along a phylogenetic tree.

This module provides two simulators:

1. ``simulate_alignment_numpy``: pure-NumPy simulator used in all simulation
   experiments reported in the manuscript. It correctly handles branch-specific
   Q matrices (heterogeneous selection regimes across branches) by propagating
   codon states through the tree using scipy.linalg.expm.  All nine experiment
   scripts (01–09) call this path via ``simulate_episodic_alignment_numpy``.

2. ``simulate_alignment_pyvolve``: thin wrapper around Pyvolve (Spielman &
   Wilke 2015), retained for reproducibility and external validation.  Pyvolve
   must be installed separately (``pip install pyvolve``).  Results from the
   two simulators are numerically equivalent under the same model parameters;
   the NumPy path was used for all published results because it requires no
   additional dependencies and supports fixed-seed reproducibility via
   numpy.random.Generator.

Both simulators accept a tree in standard Newick format and a callable
``q_for_branch`` that returns the generator Q to use on a given branch.
This is how episodic selection is implemented: neutral background on all
branches except a designated foreground branch, where omega is elevated.

Tree convention
---------------
Standard Newick format is parsed by the minimal built-in parser (no ete3
dependency).  Internal nodes may or may not have names; the parser assigns
stable labels by post-order traversal if names are missing.  Branch lengths
are measured in expected substitutions per codon, matching the ``scale=True``
normalisation used in ``build_gy94_generator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from scipy.linalg import expm

from .gy94 import sense_codons, build_gy94_generator


# ---------------------------------------------------------------------------
# Minimal Newick parser (no ete3 dependency)
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """Rooted phylogenetic tree node."""
    name: str = ""
    branch_length: float = 0.0
    children: list["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def iter_postorder(self):
        for child in self.children:
            yield from child.iter_postorder()
        yield self

    def iter_preorder(self):
        yield self
        for child in self.children:
            yield from child.iter_preorder()

    def leaves(self) -> list["TreeNode"]:
        return [n for n in self.iter_postorder() if n.is_leaf]

    def assign_internal_names(self, prefix: str = "N") -> None:
        """Give every unnamed internal node a unique name."""
        counter = 0
        for node in self.iter_postorder():
            if not node.is_leaf and not node.name:
                node.name = f"{prefix}{counter}"
                counter += 1


def parse_newick(newick: str) -> TreeNode:
    """
    Parse a Newick string into a TreeNode.

    Supports: nested parentheses, branch lengths after colons, internal node
    labels, trailing semicolon.  Does not support: quoted labels, comments,
    or NHX annotations.
    """
    s = newick.strip().rstrip(";").strip()
    pos = [0]

    def parse_node(parent: Optional[TreeNode]) -> TreeNode:
        node = TreeNode(parent=parent)
        if pos[0] < len(s) and s[pos[0]] == "(":
            pos[0] += 1
            node.children.append(parse_node(node))
            while pos[0] < len(s) and s[pos[0]] == ",":
                pos[0] += 1
                node.children.append(parse_node(node))
            if pos[0] >= len(s) or s[pos[0]] != ")":
                raise ValueError(f"Expected ')' at position {pos[0]}")
            pos[0] += 1
        name_chars = []
        while pos[0] < len(s) and s[pos[0]] not in ",():":
            name_chars.append(s[pos[0]])
            pos[0] += 1
        node.name = "".join(name_chars).strip()
        if pos[0] < len(s) and s[pos[0]] == ":":
            pos[0] += 1
            bl_chars = []
            while pos[0] < len(s) and s[pos[0]] not in ",()":
                bl_chars.append(s[pos[0]])
                pos[0] += 1
            node.branch_length = float("".join(bl_chars))
        return node

    root = parse_node(None)
    root.assign_internal_names()
    return root


# ---------------------------------------------------------------------------
# NumPy-native simulator (used for all published experiments)
# ---------------------------------------------------------------------------

def _sample_from_stationary(
    pi: np.ndarray, n_sites: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n_sites codon indices from the stationary distribution pi."""
    return rng.choice(len(pi), size=n_sites, p=pi)


def _evolve_one_branch(
    parent_states: np.ndarray,
    Q: np.ndarray,
    t: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Evolve ``parent_states`` along a branch of length t.

    Uses the transition probability matrix P = exp(Q * t) exactly — not a
    Gillespie approximation.  The 61x61 matrix is computed once and all
    sites are drawn in a single vectorised pass grouped by parent state.
    """
    if t <= 0:
        return parent_states.copy()
    P = np.clip(expm(Q * t), 0.0, None)
    P = P / P.sum(axis=1, keepdims=True)
    n_sites = parent_states.shape[0]
    child = np.empty(n_sites, dtype=np.int64)
    for s in range(P.shape[0]):
        mask = parent_states == s
        k = int(mask.sum())
        if k == 0:
            continue
        child[mask] = rng.choice(P.shape[1], size=k, p=P[s])
    return child


def simulate_alignment_numpy(
    tree: TreeNode,
    n_sites: int,
    q_for_branch: Callable[[TreeNode], np.ndarray],
    pi_root: np.ndarray,
    rng: np.random.Generator | int | None = None,
) -> dict[str, str]:
    """
    Simulate a codon alignment along ``tree`` using per-branch Q matrices.

    Parameters
    ----------
    tree : TreeNode
        Rooted phylogenetic tree.
    n_sites : int
        Number of codon sites.
    q_for_branch : callable
        Maps a TreeNode to the 61x61 generator Q for the branch entering that
        node.  Used to implement branch-heterogeneous selection regimes.
    pi_root : ndarray of shape (61,)
        Stationary distribution at the root.
    rng : Generator, int, or None
        Random source.  Pass a seeded Generator for reproducible results.

    Returns
    -------
    alignment : dict {leaf_name: codon_string}
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    codons   = sense_codons()
    n_codons = len(codons)
    if pi_root.shape != (n_codons,):
        raise ValueError(f"pi_root must have shape ({n_codons},)")
    if not np.isclose(pi_root.sum(), 1.0):
        raise ValueError("pi_root must sum to 1")

    states: dict[int, np.ndarray] = {}
    states[id(tree)] = _sample_from_stationary(pi_root, n_sites, rng)

    for node in tree.iter_preorder():
        if node.parent is None:
            continue
        Q = q_for_branch(node)
        states[id(node)] = _evolve_one_branch(
            states[id(node.parent)], Q, node.branch_length, rng
        )

    alignment: dict[str, str] = {}
    for leaf in tree.leaves():
        idx = states[id(leaf)]
        alignment[leaf.name] = "".join(codons[i] for i in idx)
    return alignment


# ---------------------------------------------------------------------------
# Standard episodic-selection setup used in the paper experiments
# ---------------------------------------------------------------------------

def make_branch_q_lookup(
    Q_background: np.ndarray,
    foreground_branches: set[str],
    Q_foreground: np.ndarray,
) -> Callable[[TreeNode], np.ndarray]:
    """
    Build a ``q_for_branch`` callable for the standard episodic-selection
    setup: Q_foreground on designated branches, Q_background elsewhere.
    Branch identity is determined by the descendant node name, matching
    the branch-site convention used by PAML and HyPhy.
    """
    def q_for_branch(node: TreeNode) -> np.ndarray:
        if node.name in foreground_branches:
            return Q_foreground
        return Q_background
    return q_for_branch


def simulate_episodic_alignment_numpy(
    newick: str,
    n_sites: int,
    omega_background: float = 1.0,
    omega_foreground: float = 1.0,
    foreground_branches: Optional[set[str]] = None,
    kappa: float = 2.0,
    pi: Optional[np.ndarray] = None,
    rng: np.random.Generator | int | None = None,
) -> tuple[dict[str, str], TreeNode]:
    """
    Simulate a codon alignment under the PHYLOSPECT episodic-selection setup.

    All simulation experiments in the manuscript use this function.  Neutral
    evolution on all branches is obtained by setting omega_foreground = 1.0
    and foreground_branches = None.

    Scaling convention
    ------------------
    Each Q matrix is independently normalised to unit expected substitution
    rate (scale=True).  Branch lengths are therefore interpretable as expected
    substitutions per codon under the model active on that branch.  This
    matches the convention used by Pyvolve and PAML evolver (Zhang, Nielsen &
    Yang 2005).

    Parameters
    ----------
    newick             : Newick tree string with branch lengths.
    n_sites            : number of codon sites.
    omega_background   : ω on all background branches (default 1.0 = neutral).
    omega_foreground   : ω on foreground branches (default 1.0).
    foreground_branches: set of node names designating foreground branches;
                         None means no foreground (all branches neutral).
    kappa              : transition/transversion ratio.
    pi                 : codon stationary frequencies; uniform if None.
    rng                : random Generator for reproducibility.

    Returns
    -------
    (alignment, tree)  — alignment dict and parsed tree.
    """
    tree = parse_newick(newick)
    from .gy94 import uniform_codon_frequencies
    if pi is None:
        pi = uniform_codon_frequencies()

    Q_bg = build_gy94_generator(omega=omega_background, kappa=kappa, pi=pi, scale=True)
    Q_fg = build_gy94_generator(omega=omega_foreground, kappa=kappa, pi=pi, scale=True)

    if foreground_branches is None:
        foreground_branches = set()
    q_for_branch = make_branch_q_lookup(Q_bg, foreground_branches, Q_fg)

    alignment = simulate_alignment_numpy(
        tree=tree, n_sites=n_sites,
        q_for_branch=q_for_branch, pi_root=pi, rng=rng,
    )
    return alignment, tree


# ---------------------------------------------------------------------------
# Pyvolve wrapper (optional — not used in published experiments)
# ---------------------------------------------------------------------------

def simulate_alignment_pyvolve(
    newick: str,
    n_sites: int,
    omega_background: float,
    omega_foreground: float,
    foreground_branches: Optional[set[str]] = None,
    kappa: float = 2.0,
    pi: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> dict[str, str]:
    """
    Simulate a codon alignment with Pyvolve (Spielman & Wilke 2015).

    Retained for external validation and reproducibility with a widely-used
    tool.  All simulation experiments in the manuscript used
    ``simulate_episodic_alignment_numpy`` instead, because it supports
    numpy.random.Generator for fixed-seed reproducibility and requires no
    additional installation.  Results from both simulators are numerically
    equivalent under the same model parameters.

    Pyvolve must be installed separately: ``pip install pyvolve``.
    """
    try:
        import pyvolve  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Pyvolve is required for simulate_alignment_pyvolve. "
            "Install with: pip install pyvolve"
        ) from e

    if foreground_branches is None:
        foreground_branches = set()

    tree = parse_newick(newick)
    has_foreground = len(foreground_branches) > 0
    if has_foreground:
        annotated = _write_labelled_newick(
            tree, foreground_branches, label="fg", default_label="bg"
        )
    else:
        annotated = _write_labelled_newick(tree, set(), label="fg")

    pyvolve_tree = pyvolve.read_tree(tree=annotated)

    state_freqs_arr = None
    if pi is not None:
        from .gy94 import sense_codons as _sc
        our_codons = _sc()
        if len(pi) != len(our_codons):
            raise ValueError(f"pi must have length {len(our_codons)}; got {len(pi)}")
        pyvolve_codons = _pyvolve_codon_order()
        if sorted(pyvolve_codons) != sorted(our_codons):
            raise RuntimeError(
                "Pyvolve's sense-codon set does not match ours. "
                "Check genetic code configuration."
            )
        our_map = {c: float(p) for c, p in zip(our_codons, pi)}
        state_freqs_arr = np.array(
            [our_map[c] for c in pyvolve_codons], dtype=float
        )
        state_freqs_arr = state_freqs_arr / state_freqs_arr.sum()

    def _build_params(omega: float) -> dict:
        params = {"omega": omega, "kappa": kappa}
        if state_freqs_arr is not None:
            params["state_freqs"] = state_freqs_arr
        return params

    if has_foreground:
        model_bg  = pyvolve.Model("codon", _build_params(omega_background), name="bg")
        model_fg  = pyvolve.Model("codon", _build_params(omega_foreground),  name="fg")
        partition = pyvolve.Partition(
            models=[model_bg, model_fg], size=n_sites, root_model_name="bg"
        )
    else:
        model_bg  = pyvolve.Model("codon", _build_params(omega_background))
        partition = pyvolve.Partition(models=model_bg, size=n_sites)

    evolver = pyvolve.Evolver(tree=pyvolve_tree, partitions=partition)
    if seed is not None:
        np.random.seed(seed)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        seqfile = os.path.join(tmpdir, "aln.fasta")
        try:
            evolver(seqfile=seqfile, seqfmt="fasta",
                    write_anc=False, ratefile=None, infofile=None)
        except TypeError:
            evolver(seqfile=seqfile, seqfmt="fasta")
        seqs = _read_fasta(seqfile)
    return seqs


def _read_fasta(path: str) -> dict[str, str]:
    """Minimal FASTA reader."""
    out: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            out[name] = "".join(chunks)
    return out


def _pyvolve_codon_order() -> list[str]:
    """Return Pyvolve's internal sense-codon ordering."""
    try:
        import pyvolve  # type: ignore
        from pyvolve import genetics  # type: ignore
        for attr in ("codons", "codon_list", "sense_codons"):
            order = getattr(genetics, attr, None)
            if order is not None and len(order) == 61:
                return list(order)
        G = getattr(genetics, "Genetics", None)
        if G is not None:
            g = G()
            for attr in ("codons", "codon_list", "sense_codons"):
                order = getattr(g, attr, None)
                if order is not None and len(order) == 61:
                    return list(order)
    except Exception:
        pass
    from .gy94 import sense_codons as _sc
    return sorted(_sc())


def _write_labelled_newick(
    tree: TreeNode,
    labelled_nodes: set[str],
    label: str,
    default_label: str | None = None,
) -> str:
    """
    Serialise the tree to Newick with Pyvolve branch-model labels.
    Appends ``#<label>`` to branches entering nodes in ``labelled_nodes``.
    """
    def recurse(n: TreeNode) -> str:
        if n.is_leaf:
            core = n.name
        else:
            inner = ",".join(recurse(c) for c in n.children)
            core  = f"({inner}){n.name}" if n.name else f"({inner})"
        suffix = ""
        if n.parent is not None:
            suffix += f":{n.branch_length}"
            if n.name in labelled_nodes:
                suffix += f"#{label}"
            elif default_label is not None:
                suffix += f"#{default_label}"
        return core + suffix
    return recurse(tree) + ";"
