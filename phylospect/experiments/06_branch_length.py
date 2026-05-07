"""
Experiment 06 — Branch-length robustness.

The original manuscript §3.5 claimed robustness to branch-length variation
across regimes (0.05 vs 0.10). That result was fabricated: the old pipeline
used an unused string label. This experiment delivers the real test.

Three regimes tested at omega=3.0 and omega=1.0 (calibration):
  short    : terminal 0.05 sub/codon
  standard : terminal 0.08 sub/codon (main analysis)
  long     : terminal 0.15 sub/codon

Output: data/06_branch_length.csv
Usage : python experiments/06_branch_length.py   (~30-45 min)
"""
from __future__ import annotations
import sys, pathlib, time
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
from phylospect.simulate import simulate_episodic_alignment_numpy
from phylospect.pipeline import run_phylospect

BRANCH_REGIMES = {
    "short":    "((A:0.05,B:0.05)AB:0.025,(C:0.05,D:0.05)CD:0.025);",
    "standard": "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);",
    "long":     "((A:0.15,B:0.15)AB:0.075,(C:0.15,D:0.15)CD:0.075);",
}
OMEGA_VALUES = [1.0, 3.0]
N_SITES      = 1000
N_REP        = 20
N_PERM       = 50
N_SM         = 20
N_SM_NULL    = 8
ALPHA        = 0.05
SEED         = 888
FG_BRANCH    = "A"
BRANCHES     = ["A","B","C","D"]

print("=" * 65)
print("EXPERIMENT 06 — Branch-length robustness")
print(f"  regimes={list(BRANCH_REGIMES.keys())}")
print(f"  omega={OMEGA_VALUES}  n_sites={N_SITES}  n_rep={N_REP}")
print("=" * 65)

rng = np.random.default_rng(SEED)
pi  = uniform_codon_frequencies()
Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)

rows = []
total = len(BRANCH_REGIMES) * len(OMEGA_VALUES)
cond  = 0

for regime, newick in BRANCH_REGIMES.items():
    for omega in OMEGA_VALUES:
        cond += 1
        t0 = time.time()
        label = "neutral" if omega == 1.0 else f"omega={omega}"
        print(f"[{cond}/{total}] {regime:<9}  {label} ...", end="", flush=True)

        sig_A, sig_ctrl = [], []
        for rep in range(N_REP):
            aln, _ = simulate_episodic_alignment_numpy(
                newick, N_SITES, 1.0, omega,
                foreground_branches={FG_BRANCH} if omega > 1 else None,
                kappa=2.0, pi=pi,
                rng=np.random.default_rng(rng.integers(1_000_000)))
            res = run_phylospect(
                newick, aln, Q0=Q0, n_perm=N_PERM,
                n_sm_samples=N_SM, n_sm_samples_null=N_SM_NULL,
                focal_branches=BRANCHES,
                rng=np.random.default_rng(rng.integers(1_000_000)))
            for b in BRANCHES:
                rows.append({
                    "regime": regime, "omega": omega, "rep": rep,
                    "branch": b, "NOE": res[b]["NOE"],
                    "pvalue": res[b]["pvalue"],
                    "is_fg": (b == FG_BRANCH and omega > 1.0),
                })
            sig_A.append(res[FG_BRANCH]["pvalue"] <= ALPHA)
            for b in ["B","C","D"]:
                sig_ctrl.append(res[b]["pvalue"] <= ALPHA)

        metric = np.mean(sig_A)
        fpr    = np.mean(sig_ctrl)
        tag    = "power" if omega > 1.0 else "FPR "
        print(f"  {tag}(A)={metric:.2f}  ctrl_FPR={fpr:.3f}  [{time.time()-t0:.0f}s]")

df = pd.DataFrame(rows)
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "06_branch_length.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print()
print("Calibration (omega=1, all branches):")
print(df[df["omega"]==1.0].groupby("regime")["pvalue"].agg(
    FPR=lambda x: (x<=ALPHA).mean(), mean_pval="mean").round(3))
print()
print("Power (omega=3, branch A):")
print(df[(df["omega"]==3.0) & (df["branch"]==FG_BRANCH)].groupby("regime")["pvalue"].agg(
    power=lambda x: (x<=ALPHA).mean()).round(2))
