# PHYLOSPECT Quickstart

This vignette walks through a complete PHYLOSPECT analysis — from a FASTA
alignment and Newick tree to a per-branch NOE p-value table — in approximately
10 minutes on a standard laptop.

---

## What you need

- Python ≥ 3.10
- PHYLOSPECT installed (`pip install -e .` from the repo root)
- The example files in this directory:
  - `example_alignment.fasta` — 4 taxa, 200 codons
  - The Newick tree string below

---

## Step 1 — Load the alignment

```python
def read_fasta(path):
    """Read a FASTA file into a dict {name: codon_string}."""
    aln = {}
    name = None
    chunks = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name:
                    aln[name] = ''.join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name:
            aln[name] = ''.join(chunks)
    return aln

aln = read_fasta('vignette/example_alignment.fasta')
print(f"Loaded {len(aln)} taxa, {len(list(aln.values())[0]) // 3} codons each")
```

Expected output:
```
Loaded 4 taxa, 200 codons each
```

---

## Step 2 — Define the tree

The example uses a four-taxon symmetric tree with typical vertebrate branch
lengths (t = 0.08 substitutions per codon site per lineage):

```python
newick = "((Taxon_A:0.08,Taxon_B:0.08)AB:0.04,(Taxon_C:0.08,Taxon_D:0.08)CD:0.04);"
```

---

## Step 3 — Build the GY94 background generator

```python
from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies

pi = uniform_codon_frequencies()   # uniform π_j = 1/61 for all sense codons
Q0 = build_gy94_generator(
    omega=1.0,    # background ω (neutral)
    kappa=2.0,    # transition/transversion ratio
    pi=pi,
    scale=True    # scale so expected substitutions per codon = 1
)
print(f"Q0 shape: {Q0.shape}")   # (61, 61)
```

---

## Step 4 — Run PHYLOSPECT

```python
import numpy as np
from phylospect.pipeline import run_phylospect

results = run_phylospect(
    newick   = newick,
    alignment = aln,
    Q0        = Q0,
    n_perm    = 100,    # bootstrap replicates (use 50 for speed, 500+ for publication)
    n_sm      = 20,     # stochastic mapping samples per observed alignment
    rng       = np.random.default_rng(42)
)
```

This will take approximately 2–5 minutes for n_perm = 100 on this small
alignment. For large-scale screening use n_perm = 50.

---

## Step 5 — Inspect the results

`run_phylospect` returns a dictionary keyed by branch name:

```python
print(f"\n{'Branch':<12} {'NOE':>10} {'p-value':>10} {'Significant':>12}")
print('-' * 46)
for branch, r in sorted(results.items()):
    sig = '*' if r['pvalue'] <= 0.05 else ''
    print(f"{branch:<12} {r['NOE']:>10.4f} {r['pvalue']:>10.3f} {sig:>12}")
```

Example output (neutral alignment — no significant branches expected):
```
Branch            NOE    p-value  Significant
----------------------------------------------
A           -0.0023      0.612
AB          -0.0011      0.701
B            0.0031      0.428
CD           0.0008      0.543
C           -0.0019      0.587
D            0.0014      0.491
```

---

## Step 6 — Save results to CSV

```python
import csv

with open('results/quickstart_results.csv', 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=['branch', 'NOE', 'pvalue'])
    writer.writeheader()
    for branch, r in sorted(results.items()):
        writer.writerow({'branch': branch, 'NOE': r['NOE'], 'pvalue': r['pvalue']})

print("Results saved to results/quickstart_results.csv")
```

---

## Interpreting the output

| Field   | Meaning |
|---------|---------|
| `NOE`   | Signed mean excess of nonsynonymous substitution rates on the branch relative to the pooled background. Positive = elevated (consistent with positive selection). Negative = suppressed (consistent with strong purifying selection). |
| `pvalue` | Fraction of parametric bootstrap null replicates with NOE ≥ observed. Small p-value (≤ 0.05) suggests episodic positive selection on that branch. |

**Practical guidance:**

- Use PHYLOSPECT as a **pre-screen**: flag branches with p ≤ 0.05 and follow
  up with PAML branch-site Model A for confirmatory site-level analysis.
- At short alignment lengths (< 500 codons) or for weak selection (ω ≈ 1.25),
  interpret p-values near 0.05 as orientational rather than conclusive.
- All tested branches share the same set of bootstrap replicates; adjust for
  multiple testing (e.g. Benjamini–Hochberg) when testing many branches.
- For genome-scale scans, use `n_perm = 50`. For individual candidate
  confirmation, use `n_perm = 500`.

---

## Full pipeline at a glance

```python
import numpy as np
from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
from phylospect.pipeline import run_phylospect

# 1. Load
aln = read_fasta('vignette/example_alignment.fasta')
newick = "((Taxon_A:0.08,Taxon_B:0.08)AB:0.04,(Taxon_C:0.08,Taxon_D:0.08)CD:0.04);"

# 2. Background model
pi = uniform_codon_frequencies()
Q0 = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)

# 3. Run
results = run_phylospect(newick, aln, Q0, n_perm=100,
                         rng=np.random.default_rng(42))

# 4. Report
for branch, r in sorted(results.items()):
    print(f"{branch}: NOE={r['NOE']:.4f}  p={r['pvalue']:.3f}")
```

---

## Further reading

- Full simulation benchmarks: `experiments/01_calibration.py` through `09_empirical_v4.py`
- Method description: manuscript Methods Section 2
- Calibration and power results: manuscript Results Sections 3.1–3.3
- Comparison with PAML: manuscript Results Section 3.4
- MNM robustness: manuscript Results Section 3.5
