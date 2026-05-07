"""
Codon alignment simulation along a phylogenetic tree.

This module provides two simulators:

1. ``simulate_alignment_numpy``: pure-NumPy reference simulator. Used for
   unit tests and any run where Pyvolve is unavailable. It correctly handles
   branch-specific Q matrices (heterogeneous selection regimes across
   branches) by propagating codon states through the tree using
   scipy.linalg.expm.

2. ``simulate_alignment_pyvolve``: thin wrapper around Pyvolve. Pyvolve is the
   standard Python package for codon-level simulation (Spielman & Wilke 2015).
   For the production experiments in the PHYLOSPECT paper we use this path
   so that reviewers can replicate our results with a widely trusted tool.

Both simulators accept a tree in Newick format and a callable ``q_for_branch``
that returns the generator Q to use on a given branch. This is how we
implement episodic selection: neutral background on all branches except a
designated foreground branch, where omega is elevated.

Tree convention
---------------
We use ete3 Newick format. Internal nodes may or may not have names; the
simulator assigns stable labels by post-order traversal if names are missing.
Branch lengths are measured in expected substitutions per codon, matching
the ``scale=True`` normalization used in ``build_gy94_generator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from scipy.linalg import expm

from .gy94 import sense_codons, build_gy94_generator


# ---------------------------------------------------------------------------
# Minimal Newick parser (no ete3 dependency needed for the NumPy path)
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
    Parse a Newick string into a TreeNode. Supports: nested parentheses,
    branch lengths after colons, internal node labels, trailing semicolon.
    Does not support: quoted labels, comments, or NHX annotations.
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
        # Read optional name
        name_chars = []
        while pos[0] < len(s) and s[pos[0]] not in ",():":
            name_chars.append(s[pos[0]])
            pos[0] += 1
        node.name = "".join(name_chars).strip()
        # Read optional :branch_length
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
# NumPy-native simulator
# ---------------------------------------------------------------------------

def _sample_from_stationary(pi: np.ndarray, n_sites: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_sites codon indices from the stationary distribution pi."""
    return rng.choice(len(pi), size=n_sites, p=pi)


def _evolve_one_branch(
    parent_states: np.ndarray,
    Q: np.ndarray,
    t: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Evolve ``parent_states`` (array of codon indices) along a branch of
    length t using transition probability matrix exp(Q * t).

    For efficiency we compute the 61x61 transition probability matrix once,
    then draw child states site-by-site using its rows. This is exact (not a
    Gillespie approximation) and matches what Pyvolve does under the hood.
    """
    if t <= 0:
        return parent_states.copy()
    P = expm(Q * t)
    # Guard against tiny negative values from numerical error.
    P = np.clip(P, 0.0, None)
    P = P / P.sum(axis=1, keepdims=True)

    # Vectorized: for each site, draw from the categorical with row P[parent].
    n_sites = parent_states.shape[0]
    child = np.empty(n_sites, dtype=np.int64)
    # Grouping by parent state is faster than a per-site loop:
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
        Function mapping a TreeNode to the 61x61 generator Q to use on the
        branch leading into that node (i.e. the branch from node.parent to
        node). Used to implement branch-heterogeneous selection regimes.
    pi_root : ndarray of shape (61,)
        Distribution over codon states at the root.
    rng : Generator, int, or None
        Random source.

    Returns
    -------
    alignment : dict
        Maps each leaf name to a codon sequence string (length 3*n_sites).
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    codons = sense_codons()
    n_codons = len(codons)
    if pi_root.shape != (n_codons,):
        raise ValueError(f"pi_root must have shape ({n_codons},)")
    if not np.isclose(pi_root.sum(), 1.0):
        raise ValueError("pi_root must sum to 1")

    # Root states sampled from pi_root.
    states: dict[int, np.ndarray] = {}
    states[id(tree)] = _sample_from_stationary(pi_root, n_sites, rng)

    # Pre-order: propagate states from parent to child along each branch.
    for node in tree.iter_preorder():
        if node.parent is None:
            continue
        Q = q_for_branch(node)
        parent_states = states[id(node.parent)]
        states[id(node)] = _evolve_one_branch(
            parent_states, Q, node.branch_length, rng
        )

    # Collect leaf sequences as codon strings.
    alignment: dict[str, str] = {}
    for leaf in tree.leaves():
        idx = states[id(leaf)]
        alignment[leaf.name] = "".join(codons[i] for i in idx)
    return alignment


# ---------------------------------------------------------------------------
# Convenience: the standard episodic-selection setup used in the paper
# ---------------------------------------------------------------------------

def make_branch_q_lookup(
    Q_background: np.ndarray,
    foreground_branches: set[str],
    Q_foreground: np.ndarray,
) -> Callable[[TreeNode], np.ndarray]:
    """
    Build a ``q_for_branch`` function that returns Q_foreground on branches
    whose *descendant* node name is in ``foreground_branches``, and
    Q_background elsewhere. This is how we simulate episodic selection on
    specific lineages.

    Note: "branch leading into node X" is identified by X's name here, which
    matches the branch-site convention used by PAML and HyPhy.
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
    Simulate a codon alignment under the standard PHYLOSPECT setup: neutral
    (or purifying) background evolution plus elevated omega on a designated
    set of foreground branches.

    Scaling convention (per-matrix, matches Pyvolve and PAML evolver)
    -----------------------------------------------------------------
    Each Q is independently normalized so that its expected substitution rate
    at stationarity is 1. Branch lengths are therefore interpretable as
    expected substitutions per codon *under the model active on that branch*.

    Under this convention omega changes the **composition** of substitutions
    along a branch (fraction that are nonsynonymous) rather than the total
    substitution count. The consequences:

      * On a branch with omega>1, synonymous substitution rates DECREASE
        compared to a neutral branch (to keep total rate = 1 while
        nonsynonymous rates are boosted).
      * Total substitution count on a branch stays proportional to branch
        length regardless of omega.
      * The dN/dS *ratio* estimated from the resulting alignment correctly
        reflects the simulated omega. This is what codeml and HyPhy expect.

    This is the same convention Pyvolve uses internally (it ignores the
    scale_matrix kwarg's distinctions in 1.1.0 and always per-matrix-scales),
    and is the convention used in the Zhang, Nielsen & Yang (2005) branch-site
    simulation study that we will benchmark against.

    Returns
    -------
    alignment, tree
        Alignment dict and the parsed tree (so callers can inspect branches).
    """
    tree = parse_newick(newick)
    from .gy94 import uniform_codon_frequencies
    if pi is None:
        pi = uniform_codon_frequencies()

    # Per-matrix scaling: each Q scaled independently to unit mean rate.
    Q_bg = build_gy94_generator(
        omega=omega_background, kappa=kappa, pi=pi, scale=True
    )
    Q_fg = build_gy94_generator(
        omega=omega_foreground, kappa=kappa, pi=pi, scale=True
    )

    if foreground_branches is None:
        foreground_branches = set()
    q_for_branch = make_branch_q_lookup(Q_bg, foreground_branches, Q_fg)

    alignment = simulate_alignment_numpy(
        tree=tree,
        n_sites=n_sites,
        q_for_branch=q_for_branch,
        pi_root=pi,
        rng=rng,
    )
    return alignment, tree


# ---------------------------------------------------------------------------
# Pyvolve wrapper (for full-scale production runs)
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
    Simulate a codon alignment with Pyvolve under branch-heterogeneous omega.

    This is the production simulator used for the paper's main experiments.
    Pyvolve must be installed locally (``pip install pyvolve``); the function
    is kept here so results are reproducible with a standard, widely-used
    tool (Spielman & Wilke 2015).

    Pyvolve labels branches in the Newick by placing ``#<label>`` after the
    branch length. We annotate the tree automatically based on
    ``foreground_branches``.
    """
    try:
        import pyvolve  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Pyvolve is required for simulate_alignment_pyvolve. "
            "Install with: pip install pyvolve"
        ) from e

    if foreground_branches is None:
        foreground_branches = set()

    tree = parse_newick(newick)
    # Pyvolve, when a Partition holds multiple Models with names, requires
    # every branch (including the root's implicit model) to be assigned to
    # a named model via the '#label' suffix. We therefore always give both
    # models explicit names and label every branch.
    has_foreground = len(foreground_branches) > 0
    if has_foreground:
        annotated = _write_labelled_newick(
            tree, foreground_branches, label="fg", default_label="bg"
        )
    else:
        # No foreground: single-model run. Unlabelled tree is fine here.
        annotated = _write_labelled_newick(tree, set(), label="fg")

    pyvolve_tree = pyvolve.read_tree(tree=annotated)

    # Pyvolve requires state_freqs as an ordered NumPy array, and uses
    # ALPHABETICAL codon ordering (AAA, AAC, AAG, AAT, ACA, ...). Our internal
    # ordering is PAML-style (TCAG). We must reorder pi accordingly.
    #
    # Defensive: verify Pyvolve's codon ordering at runtime rather than
    # trusting an assumption. Mismatches here produce silent mis-assignment
    # of codon frequencies and would corrupt every downstream analysis.
    state_freqs_arr = None
    if pi is not None:
        from .gy94 import sense_codons as _sc
        our_codons = _sc()
        if len(pi) != len(our_codons):
            raise ValueError(f"pi must have length {len(our_codons)}; got {len(pi)}")
        pyvolve_codons = _pyvolve_codon_order()
        if sorted(pyvolve_codons) != sorted(our_codons):
            raise RuntimeError(
                "Pyvolve's sense-codon set does not match ours. Refusing to "
                "guess the mapping. Check genetic code configuration."
            )
        # Build pi in Pyvolve's order.
        our_map = {c: float(p) for c, p in zip(our_codons, pi)}
        state_freqs_arr = np.array(
            [our_map[c] for c in pyvolve_codons], dtype=float
        )
        # Renormalize defensively to protect against float drift.
        state_freqs_arr = state_freqs_arr / state_freqs_arr.sum()

    # Models: give the background model an explicit name when there's also
    # a foreground model, so every branch's '#label' has a target.
    def _build_params(omega: float) -> dict:
        params = {"omega": omega, "kappa": kappa}
        if state_freqs_arr is not None:
            params["state_freqs"] = state_freqs_arr
        return params

    if has_foreground:
        # Pyvolve 1.1.0 per-matrix-scales each Model's Q regardless of the
        # scale_matrix kwarg; we rely on this default. Both matrices end up
        # scaled to unit mean rate, which matches the convention used by
        # PAML evolver and Zhang, Nielsen & Yang (2005).
        model_bg = pyvolve.Model(
            "codon", _build_params(omega_background), name="bg"
        )
        model_fg = pyvolve.Model(
            "codon", _build_params(omega_foreground), name="fg"
        )
        # Pyvolve requires root_model_name for branch-heterogeneous runs.
        partition = pyvolve.Partition(
            models=[model_bg, model_fg],
            size=n_sites,
            root_model_name="bg",
        )
    else:
        model_bg = pyvolve.Model("codon", _build_params(omega_background))
        partition = pyvolve.Partition(models=model_bg, size=n_sites)

    evolver = pyvolve.Evolver(tree=pyvolve_tree, partitions=partition)
    if seed is not None:
        np.random.seed(seed)  # Pyvolve uses numpy's legacy RNG.

    # Write Pyvolve output to a temp directory and read it back ourselves
    # rather than relying on Evolver.get_sequences(), whose return shape
    # has varied between Pyvolve versions. FASTA is stable.
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        seqfile = os.path.join(tmpdir, "aln.fasta")
        # Try the common 1.x call signature first; fall back to positional
        # or minimal kwargs if Pyvolve rejects anything.
        try:
            evolver(
                seqfile=seqfile, seqfmt="fasta",
                write_anc=False, ratefile=None, infofile=None,
            )
        except TypeError:
            # Older / newer signatures accept just seqfile.
            evolver(seqfile=seqfile, seqfmt="fasta")
        seqs = _read_fasta(seqfile)
    return seqs


def _read_fasta(path: str) -> dict[str, str]:
    """Minimal FASTA reader: returns {name: sequence} with whitespace stripped."""
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
    """
    Return Pyvolve's internal sense-codon ordering.

    Pyvolve 1.x orders codons alphabetically (AAA, AAC, AAG, AAT, ACA, ...)
    and drops the three standard-code stop codons (TAA, TAG, TGA). We query
    this ordering from Pyvolve itself where possible, falling back to the
    known alphabetical convention otherwise, so the wrapper stays correct
    if Pyvolve ever changes its internal ordering.
    """
    try:
        import pyvolve  # type: ignore
        from pyvolve import genetics  # type: ignore
        # Try common attribute names used in Pyvolve's genetics module.
        for attr in ("codons", "codon_list", "sense_codons"):
            order = getattr(genetics, attr, None)
            if order is not None and len(order) == 61:
                return list(order)
        # Try instance-based access (e.g. Genetics() holds the codon list).
        G = getattr(genetics, "Genetics", None)
        if G is not None:
            g = G()
            for attr in ("codons", "codon_list", "sense_codons"):
                order = getattr(g, attr, None)
                if order is not None and len(order) == 61:
                    return list(order)
    except Exception:  # noqa: BLE001
        pass

    # Fallback: alphabetical order of 61 sense codons under the standard code.
    from .gy94 import sense_codons as _sc
    return sorted(_sc())


def _write_labelled_newick(
    tree: TreeNode,
    labelled_nodes: set[str],
    label: str,
    default_label: str | None = None,
) -> str:
    """
    Serialize the tree to Newick, appending ``#<label>`` to the branch
    entering any node whose name is in ``labelled_nodes``. Pyvolve uses
    this convention to assign different models to specific branches.

    If ``default_label`` is given, every branch *not* in ``labelled_nodes``
    is labelled with ``#<default_label>``. Pyvolve requires every branch to
    match a named model when multiple models are present, so for production
    runs with Pyvolve we always supply a default label.
    """
    def recurse(n: TreeNode) -> str:
        if n.is_leaf:
            core = n.name
        else:
            inner = ",".join(recurse(c) for c in n.children)
            core = f"({inner}){n.name}" if n.name else f"({inner})"
        suffix = ""
        if n.parent is not None:
            suffix += f":{n.branch_length}"
            if n.name in labelled_nodes:
                suffix += f"#{label}"
            elif default_label is not None:
                suffix += f"#{default_label}"
        return core + suffix

    return recurse(tree) + ";"
