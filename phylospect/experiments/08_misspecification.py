"""
Experiment 08 — Model misspecification robustness: multinucleotide mutations.

The GY94 codon model assumed by PAML's branch-site Model A permits only
single-nucleotide changes per substitution event. Real biology violates
this: 1-3% of substitutions involve two or three adjacent nucleotides
changing together (multinucleotide mutations, MNMs; Schrider et al. 2011,
Harris & Nielsen 2014). Venkat, Hahn & Thornton 2018 showed that this
single-nt assumption causes PAML's branch-site test to produce inflated
false positive rates on neutrally-evolving sequences when MNMs are present.

This experiment tests whether PHYLOSPECT's likelihood-free, operator-based
framework is robust to this assumption violation. Under true neutrality
with MNM contamination:
  - If PAML shows elevated FPR and PHYLOSPECT stays at nominal 5%,
    PHYLOSPECT has a real, publishable advantage on misspecified models.
  - If both inflate, PHYLOSPECT has no special advantage here.
  - If PAML stays at 5% but PHYLOSPECT inflates, we were wrong about
    likelihood-freeness giving robustness.

Simulation model
----------------
We simulate under neutral GY94 as usual, then introduce MNMs by randomly
selecting a small fraction of sites in each terminal branch's post-evolution
sequence and mutating two adjacent nucleotides at those positions.
Default MNM rate: 3% of codon sites per terminal branch (~Venkat range).

Output: data/08_misspecification.csv
Usage : python experiments/08_misspecification.py
Runtime: ~3-4 hours (20 reps × 2 MNM conditions × 2 methods).
"""

from __future__ import annotations
import sys, pathlib, os, subprocess, tempfile, re, time, shutil, string
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from phylospect.gy94 import (
    build_gy94_generator, uniform_codon_frequencies,
    sense_codons, codon_index, STANDARD_CODE,
)
from phylospect.simulate import (
    parse_newick, simulate_episodic_alignment_numpy, _write_labelled_newick,
)
from phylospect.pipeline import run_phylospect

# ─── Configuration ─────────────────────────────────────────────────────
CODEML_PATH    = r"C:\Users\iamma\phylospect\phylospect\bin\codeml.exe"
NEWICK         = "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);"
N_SITES        = 1000
N_REP          = 20
N_PERM         = 50
N_SM           = 20
N_SM_NULL      = 8
ALPHA          = 0.05
SEED           = 5555
FG_BRANCH      = "A"

# MNM rate: fraction of codon sites per terminal branch that receive a
# multinucleotide mutation (2 adjacent nucleotides changing together).
# 0 = baseline (no misspecification). 0.03 = 3% of sites, matching the
# range documented by Venkat/Hahn/Thornton 2018 for mammalian genomes.
MNM_RATES = [0.00, 0.03]


# ─── MNM contamination ─────────────────────────────────────────────────

SENSE_CODONS = set(sense_codons())
CODE = STANDARD_CODE


def _inject_mnms(aln: dict, rate: float, rng: np.random.Generator) -> dict:
    """
    Inject multinucleotide mutations into a simulated alignment.

    For each leaf sequence, a fraction `rate` of codon sites are selected
    uniformly at random; at each selected site, two adjacent nucleotides
    are simultaneously mutated to random different nucleotides. Any
    mutation that produces a stop codon is redrawn up to 5 times; if it
    cannot be resolved, that site is left unchanged.

    This models double-nt substitutions invisible under GY94's single-nt
    restriction. It does NOT affect PHYLOSPECT's null simulation (which
    also uses pure GY94), so the null-hypothesis structure is preserved:
    we are injecting "noise" that violates the GY94 assumption without
    adding selection.

    Parameters
    ----------
    aln : dict {leaf_name: nucleotide_string}
    rate : fraction of codon sites per leaf to receive an MNM
    rng : random Generator

    Returns
    -------
    Contaminated alignment (new dict, same keys).
    """
    nt_set = ("T", "C", "A", "G")
    out = {}
    for name, seq in aln.items():
        if rate <= 0:
            out[name] = seq
            continue
        seq_list = list(seq)
        n_codons = len(seq) // 3
        n_mnm = int(round(rate * n_codons))
        if n_mnm <= 0:
            out[name] = seq
            continue
        sites = rng.choice(n_codons, size=n_mnm, replace=False)
        for s in sites:
            base = 3 * s
            codon_orig = "".join(seq_list[base:base + 3])
            # Pick which adjacent pair of positions to mutate (0-1 or 1-2).
            for _ in range(5):
                pair_start = int(rng.integers(2))  # 0 or 1
                new_nts = rng.choice(nt_set, size=2, replace=True)
                new_codon = list(codon_orig)
                new_codon[pair_start]     = new_nts[0]
                new_codon[pair_start + 1] = new_nts[1]
                new_codon_str = "".join(new_codon)
                if new_codon_str in SENSE_CODONS and new_codon_str != codon_orig:
                    seq_list[base:base + 3] = list(new_codon_str)
                    break
        out[name] = "".join(seq_list)
    return out


# ─── PAML helpers (fixed codeml) ──────────────────────────────────────

def _aln_to_phylip(aln, n_sites):
    lines = [f" {len(aln)}  {3 * n_sites}"]
    for name, seq in aln.items():
        lines.append(f"{name:<10}{seq}")
    return "\n".join(lines) + "\n"


def _write_paml_tree(newick, fg_branch):
    tree = parse_newick(newick)
    labelled = _write_labelled_newick(tree, labelled_nodes={fg_branch}, label="1")
    labelled = re.sub(r"\)([A-Za-z0-9_]+);\s*$", ");", labelled)
    leaves = [n.name for n in tree.leaves()]
    return f" {len(leaves)}  1\n{labelled}\n"


def _ctl(seqfile, treefile, outfile, fix_omega, omega):
    return (
        f"      seqfile = {seqfile}\n"
        f"     treefile = {treefile}\n"
        f"      outfile = {outfile}\n"
        f"       noisy = 3\n     verbose = 0\n     runmode = 0\n"
        f"     seqtype = 1\n   CodonFreq = 2\n       model = 2\n"
        f"    NSsites = 2\n       icode = 0\n       clock = 0\n"
        f"   fix_omega = {fix_omega}\n       omega = {omega}\n"
        f"   fix_kappa = 0\n       kappa = 2.0\n"
        f"   fix_alpha = 1\n       alpha = 0\n       ncatG = 3\n"
        f"      getSE = 0\n RateAncestor = 0\n"
        f"  Small_Diff = .5e-6\n   cleandata = 1\n"
        f"  fix_blength = 0\n      method = 0\n"
    )


def _parse_lnl(text):
    m = re.search(r"lnL\(ntime:.*?\):\s*(-?\d+\.\d+)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"lnL\s*=\s*(-?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def run_paml_branch_site(aln, newick, n_sites, fg_branch):
    """Returns p-value (50:50 mixture) or None on failure."""
    from scipy.stats import chi2
    with tempfile.TemporaryDirectory() as tmp:
        seq_path  = os.path.join(tmp, "aln.phy")
        tree_path = os.path.join(tmp, "tree.nwk")
        open(seq_path,  "w").write(_aln_to_phylip(aln, n_sites))
        open(tree_path, "w").write(_write_paml_tree(newick, fg_branch))
        lnls = {}
        for suffix, fix_omega, omega in [("alt", 0, 1.5), ("null", 1, 1.0)]:
            ctl = os.path.join(tmp, f"{suffix}.ctl")
            open(ctl, "w").write(_ctl("aln.phy", "tree.nwk", f"{suffix}.out",
                                       fix_omega, omega))
            try:
                subprocess.run([CODEML_PATH, os.path.basename(ctl)],
                               cwd=tmp, capture_output=True, timeout=600)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return None
            out_path = os.path.join(tmp, f"{suffix}.out")
            if not os.path.isfile(out_path):
                return None
            lnl = _parse_lnl(open(out_path, errors="replace").read())
            if lnl is None:
                return None
            lnls[suffix] = lnl
        lrt = max(0.0, 2.0 * (lnls["alt"] - lnls["null"]))
        return 0.5 * float(chi2.sf(lrt, df=1))


# ─── Main ────────────────────────────────────────────────────────────────

print("=" * 70)
print("EXPERIMENT 08 — Model misspecification robustness (MNMs)")
print("=" * 70)
print(f"  Design: under true neutrality (omega=1), inject MNMs at rate r,")
print(f"          test whether PHYLOSPECT and PAML maintain nominal FPR=5%.")
print(f"  MNM rates: {MNM_RATES}")
print(f"  n_rep: {N_REP}, n_sites: {N_SITES}")
print("=" * 70)

if not (os.path.isfile(CODEML_PATH) or shutil.which(CODEML_PATH)):
    print(f"  codeml not found at {CODEML_PATH}. Aborting.")
    sys.exit(1)

rng = np.random.default_rng(SEED)
pi  = uniform_codon_frequencies()
Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)

rows = []
for mnm_rate in MNM_RATES:
    print(f"\n--- MNM rate: {mnm_rate:.3f} "
          f"({'baseline' if mnm_rate == 0 else 'contaminated'}) ---")
    phy_pv, pal_pv = [], []
    t_start = time.time()
    for rep in range(N_REP):
        t0 = time.time()
        rng_rep = np.random.default_rng(rng.integers(1_000_000))

        # Simulate NEUTRAL alignment (omega=1 everywhere).
        aln_clean, _ = simulate_episodic_alignment_numpy(
            NEWICK, N_SITES, 1.0, 1.0,
            foreground_branches=None, kappa=2.0, pi=pi, rng=rng_rep,
        )
        # Inject MNM contamination.
        aln = _inject_mnms(aln_clean, mnm_rate, rng_rep)

        # PHYLOSPECT
        res = run_phylospect(
            NEWICK, aln, Q0=Q0, n_perm=N_PERM,
            n_sm_samples=N_SM, n_sm_samples_null=N_SM_NULL,
            focal_branches=[FG_BRANCH], rng=rng_rep,
        )
        # Prefer NOE p-value as primary (NOE primary per project decision).
        phy_p = res[FG_BRANCH].get("NOE_pvalue", res[FG_BRANCH]["pvalue"])
        phy_pv.append(phy_p)

        # PAML
        pal_p = run_paml_branch_site(aln, NEWICK, N_SITES, FG_BRANCH)
        pal_pv.append(pal_p if pal_p is not None else np.nan)

        rows.append({
            "mnm_rate": mnm_rate, "rep": rep,
            "phylospect_pvalue": phy_p,
            "paml_pvalue": pal_pv[-1],
        })
        print(f"  rep {rep+1:>2}/{N_REP}  "
              f"PHYLOSPECT p={phy_p:.3f}  PAML p={pal_pv[-1]:.3f}  "
              f"[{time.time()-t0:.0f}s]")

    phy_fpr  = np.mean(np.array(phy_pv) <= ALPHA)
    pal_arr  = np.array(pal_pv)
    pal_fpr  = np.nanmean(pal_arr <= ALPHA) if np.sum(~np.isnan(pal_arr)) else np.nan
    print(f"\n  FPR at alpha={ALPHA}:")
    print(f"    PHYLOSPECT = {phy_fpr:.3f}  (nominal: {ALPHA})")
    print(f"    PAML       = {pal_fpr:.3f}  (nominal: {ALPHA})")
    print(f"  Total time: {(time.time()-t_start)/60:.1f} min")

df = pd.DataFrame(rows)
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "08_misspecification.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved: {out}\n")

print("=" * 70)
print("FPR COMPARISON UNDER MODEL MISSPECIFICATION")
print("=" * 70)
summary = df.groupby("mnm_rate").agg(
    phylospect_FPR=("phylospect_pvalue", lambda x: (x <= ALPHA).mean()),
    paml_FPR=("paml_pvalue", lambda x: np.nanmean(x <= ALPHA)),
).round(3)
print(summary.to_string())
print()
print("Interpretation:")
print("  - Nominal FPR = 0.05 under both methods.")
print("  - MNM rate 0.00 = baseline GY94, should be calibrated for both.")
print("  - MNM rate 0.03 = MNMs injected; if PAML's FPR inflates while")
print("    PHYLOSPECT's stays near 0.05, PHYLOSPECT is robust to this")
print("    known biologically-relevant model violation (Venkat et al. 2018).")
