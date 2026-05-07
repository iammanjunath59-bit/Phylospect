"""
Goldman-Yang (1994) codon substitution model.

Reference
---------
Goldman, N., Yang, Z., 1994. A codon-based model of nucleotide substitution
for protein-coding DNA sequences. Mol. Biol. Evol. 11, 725-736.

Rate from codon i to codon j (i != j):
  - 0                          if i and j differ at >1 nucleotide position
  - pi_j                       if synonymous transversion (single-nt change)
  - kappa * pi_j               if synonymous transition
  - omega * pi_j               if nonsynonymous transversion
  - kappa * omega * pi_j       if nonsynonymous transition

Diagonal entries set so rows sum to zero (valid continuous-time Markov
generator). Stop codons (TAA, TAG, TGA under the standard code) are excluded,
giving a 61x61 sense-codon generator.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Standard genetic code and codon / amino-acid utilities
# ---------------------------------------------------------------------------

# Standard genetic code (NCBI transl_table=1).
STANDARD_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

NUCLEOTIDES = ("T", "C", "A", "G")

# Transition pairs: purine-purine (A<->G) and pyrimidine-pyrimidine (C<->T).
# Everything else between distinct nucleotides is a transversion.
_TRANSITIONS = frozenset({("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")})


def sense_codons(code: dict[str, str] = STANDARD_CODE) -> list[str]:
    """Return the 61 sense codons in canonical T,C,A,G x T,C,A,G x T,C,A,G order."""
    codons = []
    for a in NUCLEOTIDES:
        for b in NUCLEOTIDES:
            for c in NUCLEOTIDES:
                codon = a + b + c
                if code[codon] != "*":
                    codons.append(codon)
    return codons


def codon_index(code: dict[str, str] = STANDARD_CODE) -> dict[str, int]:
    """Map each sense codon to its row/column index in a 61x61 matrix."""
    return {c: i for i, c in enumerate(sense_codons(code))}


def is_transition(n1: str, n2: str) -> bool:
    """True if (n1, n2) is a transition (A<->G or C<->T)."""
    return (n1, n2) in _TRANSITIONS


def codon_diff(c1: str, c2: str) -> list[tuple[int, str, str]]:
    """
    Return list of (position, nt_in_c1, nt_in_c2) for each position where
    codons differ. Length of return value equals the number of differing
    nucleotide positions.
    """
    return [(k, c1[k], c2[k]) for k in range(3) if c1[k] != c2[k]]


# ---------------------------------------------------------------------------
# Equilibrium codon frequencies
# ---------------------------------------------------------------------------

def uniform_codon_frequencies(code: dict[str, str] = STANDARD_CODE) -> np.ndarray:
    """Uniform distribution over sense codons (61 values summing to 1)."""
    codons = sense_codons(code)
    return np.full(len(codons), 1.0 / len(codons))


def f3x4_codon_frequencies(
    nt_freqs_by_position: tuple[dict[str, float], dict[str, float], dict[str, float]],
    code: dict[str, str] = STANDARD_CODE,
) -> np.ndarray:
    """
    F3x4 codon frequencies: product of position-specific nucleotide frequencies,
    renormalized over sense codons.

    Parameters
    ----------
    nt_freqs_by_position
        Triple of dicts (pos1, pos2, pos3), each mapping 'T'/'C'/'A'/'G' -> freq.
    """
    codons = sense_codons(code)
    raw = np.empty(len(codons))
    for i, c in enumerate(codons):
        raw[i] = (
            nt_freqs_by_position[0][c[0]]
            * nt_freqs_by_position[1][c[1]]
            * nt_freqs_by_position[2][c[2]]
        )
    return raw / raw.sum()


# ---------------------------------------------------------------------------
# GY94 generator construction
# ---------------------------------------------------------------------------

def build_gy94_generator(
    omega: float = 1.0,
    kappa: float = 2.0,
    pi: np.ndarray | None = None,
    scale: bool = True,
    code: dict[str, str] = STANDARD_CODE,
) -> np.ndarray:
    """
    Build a 61x61 GY94 codon substitution generator Q.

    Parameters
    ----------
    omega : float
        dN/dS ratio. omega=1 is neutral, omega>1 positive, omega<1 purifying.
    kappa : float
        Transition / transversion rate ratio.
    pi : ndarray of shape (61,) or None
        Equilibrium codon frequencies. If None, uniform frequencies are used.
    scale : bool
        If True, scale Q so the expected number of substitutions per unit time
        is 1 under stationarity: -sum_i pi_i * Q[i, i] = 1. This makes branch
        lengths directly interpretable as expected substitutions per codon.
    code : dict
        Genetic code mapping codon -> amino acid (or '*' for stop).

    Returns
    -------
    Q : ndarray of shape (61, 61)
        Valid continuous-time Markov generator. Rows sum to zero.
    """
    codons = sense_codons(code)
    n = len(codons)
    if pi is None:
        pi = uniform_codon_frequencies(code)
    pi = np.asarray(pi, dtype=float)
    if pi.shape != (n,):
        raise ValueError(f"pi must have shape ({n},), got {pi.shape}")
    if not np.isclose(pi.sum(), 1.0):
        raise ValueError("pi must sum to 1")

    Q = np.zeros((n, n), dtype=float)

    for i, ci in enumerate(codons):
        ai = code[ci]
        for j, cj in enumerate(codons):
            if i == j:
                continue
            diff = codon_diff(ci, cj)
            if len(diff) != 1:
                # GY94 permits only single-nucleotide changes between codons.
                continue
            _, n_from, n_to = diff[0]
            aj = code[cj]

            rate = pi[j]
            if is_transition(n_from, n_to):
                rate *= kappa
            if ai != aj:  # nonsynonymous
                rate *= omega
            Q[i, j] = rate

        Q[i, i] = -Q[i, :].sum()

    if scale:
        # mu = -sum_i pi_i * Q[i,i] is expected substitutions per unit time.
        mu = -np.sum(pi * np.diag(Q))
        if mu <= 0:
            raise RuntimeError("Non-positive scaling factor; check inputs.")
        Q = Q / mu

    return Q


def validate_generator(Q: np.ndarray, tol: float = 1e-10) -> None:
    """
    Raise AssertionError if Q is not a valid continuous-time Markov generator:
      - square
      - off-diagonal entries non-negative
      - rows sum to zero
    """
    assert Q.ndim == 2 and Q.shape[0] == Q.shape[1], "Q must be square"
    n = Q.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    assert np.all(Q[off_diag_mask] >= -tol), "Off-diagonal entries must be >= 0"
    row_sums = Q.sum(axis=1)
    assert np.allclose(row_sums, 0.0, atol=tol), (
        f"Rows must sum to 0 (max abs row sum = {np.abs(row_sums).max():.3e})"
    )


# ---------------------------------------------------------------------------
# Convenience: classification of codon pairs for downstream analyses
# ---------------------------------------------------------------------------

def codon_pair_type(
    ci: str, cj: str, code: dict[str, str] = STANDARD_CODE
) -> str:
    """
    Classify an ordered codon pair (ci -> cj) into one of:
      'identical', 'multi', 'syn_tv', 'syn_ts', 'nonsyn_tv', 'nonsyn_ts'.

    'multi' means the codons differ at more than one nucleotide position
    (rate 0 under GY94).
    """
    if ci == cj:
        return "identical"
    diff = codon_diff(ci, cj)
    if len(diff) != 1:
        return "multi"
    _, n_from, n_to = diff[0]
    ts = is_transition(n_from, n_to)
    syn = code[ci] == code[cj]
    if syn and ts:
        return "syn_ts"
    if syn and not ts:
        return "syn_tv"
    if not syn and ts:
        return "nonsyn_ts"
    return "nonsyn_tv"


def nonsynonymous_single_nt_mask(
    code: dict[str, str] = STANDARD_CODE,
) -> np.ndarray:
    """
    Boolean mask (61x61) that is True for ordered codon pairs (i, j) where
    ci and cj differ at exactly one nucleotide and encode different amino acids.

    This is the set of entries that GY94 scales by omega, and the set the
    manuscript refers to when it says 'nonsynonymous single-nucleotide codon
    changes'.
    """
    codons = sense_codons(code)
    n = len(codons)
    mask = np.zeros((n, n), dtype=bool)
    for i, ci in enumerate(codons):
        for j, cj in enumerate(codons):
            if i == j:
                continue
            if codon_pair_type(ci, cj, code) in ("nonsyn_ts", "nonsyn_tv"):
                mask[i, j] = True
    return mask
