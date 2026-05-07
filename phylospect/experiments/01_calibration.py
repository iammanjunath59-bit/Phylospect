"""
Experiment 01 — Null calibration (NOE statistic, parametric bootstrap).

Tests that PHYLOSPECT p-values are uniformly distributed under neutrality.
Output: data/01_calibration.csv
Usage : python experiments/01_calibration.py   (~15-20 min)
"""
from __future__ import annotations
import sys, pathlib, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
from phylospect.simulate import simulate_episodic_alignment_numpy
from phylospect.pipeline import run_phylospect

NEWICK    = "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);"
N_SITES   = 1000
N_REP     = 30
N_PERM    = 50
N_SM      = 20
N_SM_NULL = 8
ALPHA     = 0.05
SEED      = 42
BRANCHES  = ["A","B","C","D"]

print("=" * 60)
print("EXPERIMENT 01 — Null calibration")
print(f"  n_sites={N_SITES}  n_rep={N_REP}  n_perm={N_PERM}")
print("=" * 60)

rng = np.random.default_rng(SEED)
pi  = uniform_codon_frequencies()
Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)

rows = []
t_start = time.time()

for rep in range(N_REP):
    aln, _ = simulate_episodic_alignment_numpy(
        NEWICK, N_SITES, 1.0, 1.0, kappa=2.0, pi=pi,
        rng=np.random.default_rng(rng.integers(1_000_000)))
    res = run_phylospect(
        NEWICK, aln, Q0=Q0, n_perm=N_PERM,
        n_sm_samples=N_SM, n_sm_samples_null=N_SM_NULL,
        focal_branches=BRANCHES,
        rng=np.random.default_rng(rng.integers(1_000_000)))
    for b in BRANCHES:
        r = res[b]
        rows.append({"branch":b,"rep":rep,"NOE":r["NOE"],
                     "pvalue":r["pvalue"],"ICE":r["ICE"],"ICEnorm":r["ICEnorm"]})
    elapsed = time.time() - t_start
    eta = elapsed / (rep+1) * (N_REP - rep - 1)
    print(f"  rep {rep+1:>2}/{N_REP}  "
          f"p(A)={res['A']['pvalue']:.3f}  "
          f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

df = pd.DataFrame(rows)
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "01_calibration.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)

from scipy.stats import kstest
pv = df["pvalue"].values
ks_stat, ks_pval = kstest(pv, "uniform")
fpr = (pv <= ALPHA).mean()

print()
print("=" * 60)
print("SUMMARY")
print(f"  Mean p-value : {pv.mean():.3f}  (expected ~0.500)")
print(f"  FPR @ {ALPHA}   : {fpr:.3f}  (expected ~{ALPHA})")
print(f"  KS p-value   : {ks_pval:.3f}  (pass if >=0.05)")
print("  " + ("✓ PASSED" if ks_pval >= 0.05 else "✗ WARNING"))
print(f"\nSaved: {out}")
