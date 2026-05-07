"""
Experiment 05 — Runtime benchmark: PHYLOSPECT vs PAML.

Timing per foreground-branch test. PAML was fixed in the codeml diagnostic
(correct tree format, lnL regex, Model A null); this script uses the same
corrections as 04_paml_comparison_fixed.py.

The comparison is per-branch-tested, not per-gene:
  - PHYLOSPECT: one pipeline run tests all N_TAXA terminal branches
                simultaneously (no extra cost per branch), so we report
                total_time / N_TAXA as per_branch_time.
  - PAML:       one alt + null codeml pair tests ONE branch, so its
                per_branch time equals its total time.

This is a fair per-test comparison because PAML requires a new codeml
invocation for each foreground branch you want to test, while PHYLOSPECT
computes branch-specific Qb and ΔQ for every branch in one bootstrap.

Output: data/05_runtime.csv
Usage : python experiments/05_runtime.py   (~20-40 min)
"""

from __future__ import annotations
import sys, pathlib, os, subprocess, tempfile, re, time, shutil, string
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
from phylospect.simulate import (
    parse_newick, simulate_episodic_alignment_numpy, _write_labelled_newick,
)
from phylospect.pipeline import run_phylospect

# ─── Configuration ──────────────────────────────────────────────────────
CODEML_PATH   = r"C:\Users\iamma\phylospect\phylospect\bin\codeml.exe"
N_SITES_LIST  = [500, 1000, 3000]
N_TAXA_LIST   = [4, 8]
N_REP_TIMING  = 3    # median over 3 reps per cell (keeps runtime reasonable)
SEED          = 77

NEWICKS = {
    4: "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);",
    8: ("((A:0.08,B:0.08)AB:0.02,(C:0.08,D:0.08)CD:0.02,"
        "(E:0.08,F:0.08)EF:0.02,(G:0.08,H:0.08)GH:0.02);"),
}


# ─── PAML helpers (matching the fixed diagnostic) ──────────────────────

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


def _time_paml(aln, newick, n_sites, fg_branch):
    """Time one PAML branch-site LRT (alt + null). Returns wall-clock seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        seq_path  = os.path.join(tmp, "aln.phy")
        tree_path = os.path.join(tmp, "tree.nwk")
        open(seq_path,  "w").write(_aln_to_phylip(aln, n_sites))
        open(tree_path, "w").write(_write_paml_tree(newick, fg_branch))

        t0 = time.perf_counter()
        for suffix, fix_omega, omega in [("alt", 0, 1.5), ("null", 1, 1.0)]:
            ctl = os.path.join(tmp, f"{suffix}.ctl")
            open(ctl, "w").write(_ctl("aln.phy", "tree.nwk", f"{suffix}.out",
                                       fix_omega, omega))
            subprocess.run([CODEML_PATH, os.path.basename(ctl)],
                           cwd=tmp, capture_output=True, timeout=600)
        return time.perf_counter() - t0


# ─── Main ────────────────────────────────────────────────────────────────

print("=" * 65)
print("EXPERIMENT 05 — Runtime benchmark (fixed codeml)")
print("=" * 65)

codeml_ok = os.path.isfile(CODEML_PATH) or shutil.which(CODEML_PATH) is not None
if not codeml_ok:
    print(f"  codeml not found at '{CODEML_PATH}' — aborting.")
    sys.exit(1)
print(f"  codeml: {CODEML_PATH}\n")

rng = np.random.default_rng(SEED)
pi  = uniform_codon_frequencies()
Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)

rows = []

for ntaxa in N_TAXA_LIST:
    newick = NEWICKS[ntaxa]
    leaves = list(string.ascii_uppercase[:ntaxa])

    for n_sites in N_SITES_LIST:
        # Time PHYLOSPECT (tests ALL branches in one pipeline call).
        phy_times = []
        for _ in range(N_REP_TIMING):
            aln, _ = simulate_episodic_alignment_numpy(
                newick, n_sites, 1.0, 1.5, foreground_branches={"A"},
                kappa=2.0, pi=pi,
                rng=np.random.default_rng(rng.integers(1_000_000)))
            t0 = time.perf_counter()
            run_phylospect(newick, aln, Q0=Q0, n_perm=50,
                           n_sm_samples=20, n_sm_samples_null=8,
                           focal_branches=leaves,
                           rng=np.random.default_rng(rng.integers(1_000_000)))
            phy_times.append(time.perf_counter() - t0)
        t_phy_total = float(np.median(phy_times))
        t_phy_per_br = t_phy_total / ntaxa

        # Time PAML (one foreground branch per codeml invocation).
        pal_times = []
        for _ in range(N_REP_TIMING):
            aln, _ = simulate_episodic_alignment_numpy(
                newick, n_sites, 1.0, 1.5, foreground_branches={"A"},
                kappa=2.0, pi=pi,
                rng=np.random.default_rng(rng.integers(1_000_000)))
            pal_times.append(_time_paml(aln, newick, n_sites, "A"))
        t_pal_per_br = float(np.median(pal_times))

        # Speedup: if PAML is slower, ratio > 1 means PHYLOSPECT faster.
        speedup_per_branch = t_pal_per_br / max(t_phy_per_br, 1e-6)
        # For a full N_TAXA-branch scan, PAML must run N_TAXA times; PHYLOSPECT
        # once. So the effective genome-scale speedup is:
        speedup_full_scan = (t_pal_per_br * ntaxa) / max(t_phy_total, 1e-6)

        print(f"  ntaxa={ntaxa}  n_sites={n_sites:>5}")
        print(f"    PHYLOSPECT: total={t_phy_total:6.2f}s  "
              f"per_branch={t_phy_per_br:5.2f}s")
        print(f"    PAML:       per_branch={t_pal_per_br:6.2f}s")
        print(f"    Speedup per branch: {speedup_per_branch:.2f}x  "
              f"Full {ntaxa}-branch scan: {speedup_full_scan:.2f}x")
        print()

        rows.append({
            "n_taxa": ntaxa, "n_sites": n_sites,
            "phylospect_total_s": round(t_phy_total, 3),
            "phylospect_per_branch_s": round(t_phy_per_br, 3),
            "paml_per_branch_s": round(t_pal_per_br, 3),
            "speedup_per_branch": round(speedup_per_branch, 2),
            "speedup_full_scan": round(speedup_full_scan, 2),
        })

df = pd.DataFrame(rows)
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "05_runtime.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved: {out}\n")
print(df.to_string(index=False))
print()
print("Speedup interpretation:")
print("  - 'per branch' : time to test ONE foreground branch (PAML's unit of work).")
print("  - 'full scan'  : time to test ALL branches (what phylogenomic screens require).")
print("  - For whole-genome scans: PHYLOSPECT must be compared on 'full scan' because")
print("    PAML must be invoked once per foreground assignment, while PHYLOSPECT")
print("    tests every branch in one pipeline run.")
