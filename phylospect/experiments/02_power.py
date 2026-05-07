"""
Experiment 02 — Power analysis.

Sweeps omega x n_sites x n_taxa, measuring detection power of the NOE statistic.
Directly addresses Reviewer #2 requests: omega=1.25 (fine-scale), 8-taxon trees.

Output: data/02_power.csv
Usage : python experiments/02_power.py   (~2-4 hrs depending on machine)
"""
from __future__ import annotations
import sys, pathlib, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
from phylospect.simulate import simulate_episodic_alignment_numpy
from phylospect.pipeline import run_phylospect

OMEGA_LIST  = [1.25, 1.5, 3.0]
N_SITES_LIST = [500, 1000, 3000]
N_TAXA_LIST  = [4, 8]
N_REP     = 20
N_PERM    = 50
N_SM      = 20
N_SM_NULL = 8
ALPHA     = 0.05
SEED      = 42
FG_BRANCH = "A"

NEWICKS = {
    4: "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);",
    8: ("((A:0.08,B:0.08)AB:0.02,(C:0.08,D:0.08)CD:0.02,"
        "(E:0.08,F:0.08)EF:0.02,(G:0.08,H:0.08)GH:0.02);"),
}

import string
LEAF_NAMES = {n: list(string.ascii_uppercase[:n]) for n in N_TAXA_LIST}

print("=" * 70)
print("EXPERIMENT 02 — Power analysis")
print(f"  omega={OMEGA_LIST}  n_sites={N_SITES_LIST}  n_taxa={N_TAXA_LIST}")
print(f"  n_rep={N_REP}  n_perm={N_PERM}")
print("=" * 70)

rng = np.random.default_rng(SEED)
pi  = uniform_codon_frequencies()
Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)

rows = []
total = len(N_TAXA_LIST) * len(OMEGA_LIST) * len(N_SITES_LIST)
cond  = 0
t_start = time.time()

for ntaxa, omega, n_sites in itertools.product(N_TAXA_LIST, OMEGA_LIST, N_SITES_LIST):
    cond += 1
    newick    = NEWICKS[ntaxa]
    leaves    = LEAF_NAMES[ntaxa]
    t0 = time.time()

    print(f"[{cond:>2}/{total}] ntaxa={ntaxa}  omega={omega}  n_sites={n_sites} ...",
          end="", flush=True)

    powers_A = []
    for rep in range(N_REP):
        aln, _ = simulate_episodic_alignment_numpy(
            newick, n_sites, 1.0, omega,
            foreground_branches={FG_BRANCH},
            kappa=2.0, pi=pi,
            rng=np.random.default_rng(rng.integers(1_000_000)))
        res = run_phylospect(
            newick, aln, Q0=Q0, n_perm=N_PERM,
            n_sm_samples=N_SM, n_sm_samples_null=N_SM_NULL,
            focal_branches=leaves,
            rng=np.random.default_rng(rng.integers(1_000_000)))
        for b in leaves:
            r = res[b]
            is_fg = (b == FG_BRANCH)
            rows.append({"omega":omega,"n_sites":n_sites,"n_taxa":ntaxa,
                         "rep":rep,"branch":b,"NOE":r["NOE"],
                         "pvalue":r["pvalue"],"ICEnorm":r["ICEnorm"],"is_fg":is_fg})
            if is_fg: powers_A.append(r["pvalue"] <= ALPHA)

    print(f"  power(A)={np.mean(powers_A):.2f}  [{time.time()-t0:.0f}s]")

df = pd.DataFrame(rows)
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "02_power.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)

print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
print(f"Saved: {out}")
print()
print("POWER TABLE (foreground branch A, alpha=0.05):")
df_fg = df[df["is_fg"]]
pivot = df_fg.groupby(["n_taxa","omega","n_sites"])["pvalue"].agg(
    power=lambda x: (x <= ALPHA).mean()
).reset_index().pivot_table(index=["n_taxa","omega"], columns="n_sites", values="power")
print(pivot.round(2))

print()
print("FPR TABLE (neutral branches, alpha=0.05):")
leaves_ctrl = [l for l in string.ascii_uppercase[:8] if l != FG_BRANCH]
df_ctrl = df[~df["is_fg"] & df["branch"].isin(leaves_ctrl)]
fpr_pivot = df_ctrl.groupby(["n_taxa","omega","n_sites"])["pvalue"].agg(
    fpr=lambda x: (x <= ALPHA).mean()
).reset_index().pivot_table(index=["n_taxa","omega"], columns="n_sites", values="fpr")
print(fpr_pivot.round(3))
