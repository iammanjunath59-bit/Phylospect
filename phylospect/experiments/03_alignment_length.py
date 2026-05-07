"""
Experiment 03 — Alignment length sensitivity.

Addresses Reviewer #2 #4: moved from supplementary to main text.
Find the minimum n_sites for >=50% power at each omega value.

Output: data/03_alignment_length.csv
Usage : python experiments/03_alignment_length.py   (~1-2 hrs)
"""
from __future__ import annotations
import sys, pathlib, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
from phylospect.simulate import simulate_episodic_alignment_numpy
from phylospect.pipeline import run_phylospect

NEWICK       = "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);"
OMEGA_LIST   = [1.25, 1.5, 3.0]
N_SITES_LIST = [250, 500, 1000, 3000, 5000]
N_REP        = 20
N_PERM       = 50
N_SM         = 20
N_SM_NULL    = 8
ALPHA        = 0.05
SEED         = 123
FG_BRANCH    = "A"
BRANCHES     = ["A","B","C","D"]

print("=" * 60)
print("EXPERIMENT 03 — Alignment length sensitivity")
print(f"  omega={OMEGA_LIST}")
print(f"  n_sites={N_SITES_LIST}")
print("=" * 60)

rng = np.random.default_rng(SEED)
pi  = uniform_codon_frequencies()
Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)

rows = []
total = len(OMEGA_LIST) * len(N_SITES_LIST)
cond  = 0

for omega in OMEGA_LIST:
    for n_sites in N_SITES_LIST:
        cond += 1
        t0 = time.time()
        print(f"[{cond:>2}/{total}] omega={omega}  n_sites={n_sites} ...", end="", flush=True)
        powers = []

        for rep in range(N_REP):
            aln, _ = simulate_episodic_alignment_numpy(
                NEWICK, n_sites, 1.0, omega,
                foreground_branches={FG_BRANCH},
                kappa=2.0, pi=pi,
                rng=np.random.default_rng(rng.integers(1_000_000)))
            res = run_phylospect(
                NEWICK, aln, Q0=Q0, n_perm=N_PERM,
                n_sm_samples=N_SM, n_sm_samples_null=N_SM_NULL,
                focal_branches=BRANCHES,
                rng=np.random.default_rng(rng.integers(1_000_000)))
            for b in BRANCHES:
                r = res[b]
                rows.append({"omega":omega,"n_sites":n_sites,"rep":rep,"branch":b,
                             "NOE":r["NOE"],"pvalue":r["pvalue"],"is_fg":(b==FG_BRANCH)})
            powers.append(res[FG_BRANCH]["pvalue"] <= ALPHA)

        print(f"  power={np.mean(powers):.2f}  [{time.time()-t0:.0f}s]")

df = pd.DataFrame(rows)
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "03_alignment_length.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved: {out}")

print()
print("POWER BY N_SITES AND OMEGA:")
df_fg = df[df["is_fg"]]
pivot = df_fg.groupby(["omega","n_sites"])["pvalue"].agg(
    power=lambda x: (x <= ALPHA).mean()
).reset_index().pivot(index="omega", columns="n_sites", values="power")
print(pivot.round(2))

print()
print("Minimum n_sites for >=50% power:")
for omega in OMEGA_LIST:
    row = df_fg[df_fg["omega"]==omega].groupby("n_sites")["pvalue"].agg(
        power=lambda x: (x <= ALPHA).mean())
    hits = row[row >= 0.50]
    thresh = hits.index.min() if len(hits) > 0 else f">{N_SITES_LIST[-1]}"
    print(f"  omega={omega}: {thresh} codons")
