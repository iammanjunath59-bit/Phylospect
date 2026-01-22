import numpy as np
import pandas as pd

from phylospect_core import *
from spectral_basis import learn_spectral_basis
from permutation_null import within_site_permutation_null

np.random.seed(42)

codons = 61
n_rep = 30
n_perm = 200

def simulate_Q(omega=1.0):
    Q = np.random.exponential(scale=0.01, size=(codons, codons))
    np.fill_diagonal(Q, 0)
    Q *= omega
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q

results = []

# Neutral baseline generator
Q0 = simulate_Q(1.0)

# Learn spectral basis
B = learn_spectral_basis(Q0)

for regime, omega in [
    ("neutral", 1.0),
    ("episodic_weak", 1.5),
    ("episodic_strong", 3.0)
]:
    for branch in ["normal", "long"]:
        for r in range(n_rep):

            Qpool = simulate_Q(1.0)

            if regime == "neutral":
                Qb = Qpool.copy()
            else:
                Qb = Qpool.copy()
                mask = np.random.rand(codons, codons) < 0.3
                Qb[mask] *= omega

            deltaQ = compute_deltaQ(Qb, Qpool)

            Mb_obs = projection_stat(deltaQ, B)
            ICE = signed_ICE(deltaQ)
            ICE_n = ICE_norm(deltaQ)

            null_stats = within_site_permutation_null(deltaQ, B, n_perm)

            pval = (null_stats >= Mb_obs).mean()

            results.append([
                f"{regime}_{branch}", r, Mb_obs, pval, ICE, ICE_n
            ])

df = pd.DataFrame(results,
    columns=["regime","rep","Mb","pval","ICE","ICE_norm"])

df.to_csv("data/results_phylospect.csv", index=False)

print("Saved: data/results_phylospect.csv")
