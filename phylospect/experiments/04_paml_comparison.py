"""
Experiment 04 — PHYLOSPECT vs PAML branch-site comparison (FIXED).

This replaces the earlier 04_paml_comparison.py, which silently produced
0% PAML power because of three bugs:
  1. Root name in the tree file caused codeml to emit "error: end of tree
     file" and exit with non-zero return code.
  2. lnL regex did not match codeml's actual output format
     ("lnL(ntime: ...):  value" — no '=' sign).
  3. No PAML-style tree file header line ("N_TAXA  1") before the Newick.

All three are fixed here. Output: data/04_paml_comparison.csv

Usage: python experiments/04_paml_comparison_fixed.py
"""

from __future__ import annotations
import sys, os, re, subprocess, tempfile, time, itertools, shutil, pathlib, string
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
from phylospect.simulate import (
    parse_newick, simulate_episodic_alignment_numpy, _write_labelled_newick,
)
from phylospect.pipeline import run_phylospect

# ─── Configuration ──────────────────────────────────────────────────────
CODEML_PATH  = r"C:\Users\iamma\phylospect\phylospect\bin\codeml.exe"
OMEGA_VALUES = [1.5, 3.0]
N_SITES_LIST = [500, 1000, 3000]
N_TAXA_LIST  = [4]
N_REP        = 20
N_PERM       = 50
N_SM         = 20
N_SM_NULL    = 8
ALPHA        = 0.05
SEED         = 999
FG_BRANCH    = "A"

NEWICKS = {
    4: "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);",
}


# ─── PAML helpers (fixed) ───────────────────────────────────────────────

def aln_to_phylip(aln: dict, n_sites: int) -> str:
    lines = [f" {len(aln)}  {3 * n_sites}"]
    for name, seq in aln.items():
        lines.append(f"{name:<10}{seq}")
    return "\n".join(lines) + "\n"


def write_paml_tree(newick: str, fg_branch: str) -> str:
    """
    PAML-compatible tree file:
      - Header line 'N_TAXA  1' as required by codeml.
      - Newick without root label (codeml chokes on it).
      - Foreground branch marked with #1 (PAML branch-site convention).
    """
    tree = parse_newick(newick)
    labelled = _write_labelled_newick(
        tree, labelled_nodes={fg_branch}, label="1"
    )
    # Strip trailing root name emitted by _write_labelled_newick.
    labelled = re.sub(r"\)([A-Za-z0-9_]+);\s*$", ");", labelled)
    leaves = [n.name for n in tree.leaves()]
    return f" {len(leaves)}  1\n{labelled}\n"


def _base_ctl(seqfile, treefile, outfile, fix_omega, omega):
    return (
        f"      seqfile = {seqfile}\n"
        f"     treefile = {treefile}\n"
        f"      outfile = {outfile}\n"
        f"       noisy = 3\n"
        f"     verbose = 0\n"
        f"     runmode = 0\n"
        f"     seqtype = 1\n"
        f"   CodonFreq = 2\n"
        f"       model = 2\n"
        f"    NSsites = 2\n"
        f"       icode = 0\n"
        f"       clock = 0\n"
        f"   fix_omega = {fix_omega}\n"
        f"       omega = {omega}\n"
        f"   fix_kappa = 0\n"
        f"       kappa = 2.0\n"
        f"   fix_alpha = 1\n"
        f"       alpha = 0\n"
        f"       ncatG = 3\n"
        f"      getSE = 0\n"
        f" RateAncestor = 0\n"
        f"  Small_Diff = .5e-6\n"
        f"   cleandata = 1\n"
        f"  fix_blength = 0\n"
        f"      method = 0\n"
    )


def parse_lnl(text: str):
    """Match codeml's actual output format."""
    m = re.search(r"lnL\(ntime:.*?\):\s*(-?\d+\.\d+)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"lnL\s*=\s*(-?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def run_codeml_branch_site(aln, newick, n_sites, fg_branch, tmpdir):
    """
    Run PAML branch-site Model A LRT.
    Returns p-value (50:50 mixture following Yang et al. 2005), or None on failure.
    """
    from scipy.stats import chi2

    # Write PHYLIP alignment and PAML tree.
    seq_path  = os.path.join(tmpdir, "aln.phy")
    tree_path = os.path.join(tmpdir, "tree.nwk")
    with open(seq_path, "w") as fh:
        fh.write(aln_to_phylip(aln, n_sites))
    with open(tree_path, "w") as fh:
        fh.write(write_paml_tree(newick, fg_branch))

    # Run alt + null.
    lnls = {}
    for suffix, fix_omega, omega_val in [("alt", 0, 1.5), ("null", 1, 1.0)]:
        out_path = os.path.join(tmpdir, f"{suffix}.out")
        ctl_path = os.path.join(tmpdir, f"{suffix}.ctl")
        with open(ctl_path, "w") as fh:
            fh.write(_base_ctl(
                "aln.phy", "tree.nwk", f"{suffix}.out",
                fix_omega, omega_val,
            ))
        try:
            subprocess.run(
                [CODEML_PATH, ctl_path.split(os.sep)[-1]],
                cwd=tmpdir, capture_output=True, timeout=600,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if not os.path.isfile(out_path):
            return None
        with open(out_path, errors="replace") as fh:
            lnl = parse_lnl(fh.read())
        if lnl is None:
            return None
        lnls[suffix] = lnl

    lrt = max(0.0, 2.0 * (lnls["alt"] - lnls["null"]))
    # 50:50 mixture null distribution (Yang & Nielsen 2005, MBE).
    return 0.5 * float(chi2.sf(lrt, df=1))


# ─── Main experiment ─────────────────────────────────────────────────────

print("=" * 70)
print("EXPERIMENT 04 — PHYLOSPECT vs PAML branch-site (FIXED)")
print("=" * 70)

if not os.path.isfile(CODEML_PATH):
    alt = shutil.which("codeml")
    if alt is None:
        print(f"  ABORT: codeml not found at {CODEML_PATH}")
        sys.exit(1)
    CODEML_PATH = alt
print(f"  codeml: {CODEML_PATH}\n")

rng = np.random.default_rng(SEED)
pi  = uniform_codon_frequencies()
Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
LEAF_NAMES = {n: list(string.ascii_uppercase[:n]) for n in N_TAXA_LIST}

rows = []
total = len(N_TAXA_LIST) * len(OMEGA_VALUES) * len(N_SITES_LIST)
cond  = 0

for ntaxa, omega, n_sites in itertools.product(N_TAXA_LIST, OMEGA_VALUES, N_SITES_LIST):
    cond += 1
    newick = NEWICKS[ntaxa]
    t0 = time.time()
    print(f"[{cond:>2}/{total}] ntaxa={ntaxa}  omega={omega}  n_sites={n_sites} ...",
          end="", flush=True)

    phy_pv, pal_pv = [], []
    paml_fails = 0

    for rep in range(N_REP):
        aln, _ = simulate_episodic_alignment_numpy(
            newick, n_sites, 1.0, omega,
            foreground_branches={FG_BRANCH},
            kappa=2.0, pi=pi,
            rng=np.random.default_rng(rng.integers(1_000_000)),
        )

        # PHYLOSPECT
        res = run_phylospect(
            newick, aln, Q0=Q0, n_perm=N_PERM,
            n_sm_samples=N_SM, n_sm_samples_null=N_SM_NULL,
            focal_branches=[FG_BRANCH],
            rng=np.random.default_rng(rng.integers(1_000_000)),
        )
        phy_p = res[FG_BRANCH]["pvalue"]
        phy_pv.append(phy_p)

        # PAML
        with tempfile.TemporaryDirectory() as tmp:
            paml_p = run_codeml_branch_site(aln, newick, n_sites, FG_BRANCH, tmp)
        if paml_p is None:
            paml_fails += 1
            paml_p = np.nan
        pal_pv.append(paml_p)

        rows.append({
            "omega": omega, "n_sites": n_sites, "n_taxa": ntaxa, "rep": rep,
            "phylospect_pvalue": phy_p,
            "paml_pvalue": paml_p,
        })

    phy_pow  = np.mean(np.array(phy_pv) <= ALPHA)
    paml_pv_arr = np.array(pal_pv)
    n_valid_paml = np.sum(~np.isnan(paml_pv_arr))
    paml_pow = (
        np.nanmean(paml_pv_arr <= ALPHA) if n_valid_paml > 0 else np.nan
    )
    print(f"  PHYLOSPECT={phy_pow:.2f}  PAML={paml_pow:.2f} "
          f"({n_valid_paml}/{N_REP} valid, {paml_fails} fails)  "
          f"[{time.time()-t0:.0f}s]")

# ─── Save & summarise ────────────────────────────────────────────────────

df = pd.DataFrame(rows)
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "04_paml_comparison.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved: {out}\n")

print("=" * 70)
print("POWER COMPARISON (foreground branch A, alpha=0.05)")
print("=" * 70)
summary = df.groupby(["omega", "n_sites"]).agg(
    phylospect_power=("phylospect_pvalue", lambda x: (x <= ALPHA).mean()),
    paml_power=("paml_pvalue", lambda x: np.nanmean(x <= ALPHA)),
    paml_n_valid=("paml_pvalue", lambda x: x.notna().sum()),
).round(2)
print(summary.to_string())
print()
print("Context for interpretation:")
print("  - This is whole-branch omega elevation, the easiest regime for")
print("    branch-site LRT. PAML is near its theoretical ceiling here.")
print("  - Frame PHYLOSPECT as complementary to PAML, emphasising its")
print("    well-calibrated null, speed, and applicability without likelihood")
print("    optimisation.")
