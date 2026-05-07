"""
generate_archived_simulations.py
=================================
Generates the two representative simulated alignments that are archived in
data/simulated/ for reproducibility (as required by Reviewer #1, Point 1.4).

  01_calibration_seed42.fasta   — neutral alignment (omega=1 everywhere)
                                   used to validate null calibration
  02_power_omega3_n1000_seed0.fasta — episodic selection alignment
                                       (omega=3.0, foreground branch A,
                                        n=1000 codons) used to demonstrate
                                        detection power

Both files are generated with fixed seeds so any reader can reproduce
the exact alignment and verify that running run_phylospect() on it
gives a p-value consistent with the published calibration / power results.

Usage:
    cd C:\\Users\\iamma\\phylospect
    python scripts/generate_archived_simulations.py

Output:
    data/simulated/01_calibration_seed42.fasta
    data/simulated/01_calibration_seed42_tree.nwk
    data/simulated/02_power_omega3_n1000_seed0.fasta
    data/simulated/02_power_omega3_n1000_seed0_tree.nwk
    data/simulated/README.txt   (documents how to reproduce)
"""

from __future__ import annotations
import sys, pathlib
import numpy as np

# ── repo root ─────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
from phylospect.simulate import simulate_episodic_alignment_numpy

OUT_DIR = ROOT / "data" / "simulated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── shared parameters ─────────────────────────────────────────────────────────
NEWICK    = "((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);"
KAPPA     = 2.0
pi        = uniform_codon_frequencies()
Q0        = build_gy94_generator(omega=1.0, kappa=KAPPA, pi=pi, scale=True)


def write_fasta(path: pathlib.Path, aln: dict[str, str]) -> None:
    with open(path, "w") as fh:
        for name, seq in aln.items():
            fh.write(f">{name}\n{seq}\n")
    print(f"  Written: {path}")


def write_tree(path: pathlib.Path, newick: str) -> None:
    path.write_text(newick.strip() + "\n")
    print(f"  Written: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# File 1: null calibration replicate (seed 42)
# This is a single neutral alignment that, when run through run_phylospect(),
# should give NOE p-values uniformly distributed — consistent with calibrated
# null behaviour (Experiment 01, KS p = 0.815, mean p = 0.494).
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 01_calibration_seed42.fasta ...")
rng_calib = np.random.default_rng(42)

aln_null, _ = simulate_episodic_alignment_numpy(
    newick           = NEWICK,
    n_sites          = 1000,
    omega_background = 1.0,
    omega_foreground = 1.0,   # neutral everywhere — this IS the null
    foreground_branches = None,
    kappa            = KAPPA,
    pi               = pi,
    rng              = rng_calib,
)

write_fasta(OUT_DIR / "01_calibration_seed42.fasta", aln_null)
write_tree(OUT_DIR / "01_calibration_seed42_tree.nwk", NEWICK)


# ─────────────────────────────────────────────────────────────────────────────
# File 2: power demonstration replicate (seed 0, omega=3.0, n=1000)
# This alignment has episodic positive selection on branch A (omega_fg=3.0).
# When run through run_phylospect(), branch A should give a significantly
# small NOE p-value, consistent with Experiment 02 power results
# (power = 50% at omega=3.0, n=1000 on 4-taxon tree).
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating 02_power_omega3_n1000_seed0.fasta ...")
rng_power = np.random.default_rng(0)

aln_power, _ = simulate_episodic_alignment_numpy(
    newick              = NEWICK,
    n_sites             = 1000,
    omega_background    = 1.0,
    omega_foreground    = 3.0,
    foreground_branches = {"A"},   # branch A is under selection
    kappa               = KAPPA,
    pi                  = pi,
    rng                 = rng_power,
)

write_fasta(OUT_DIR / "02_power_omega3_n1000_seed0.fasta", aln_power)
write_tree(OUT_DIR / "02_power_omega3_n1000_seed0_tree.nwk", NEWICK)


# ─────────────────────────────────────────────────────────────────────────────
# README for the simulated/ directory
# ─────────────────────────────────────────────────────────────────────────────
readme_text = """\
data/simulated/ — Representative archived simulation outputs
=============================================================

These two files provide seed-fixed, reproducible simulated codon alignments
that allow any user to verify PHYLOSPECT's calibration and power claims
without running the full overnight simulation benchmarks.

FILES
-----

01_calibration_seed42.fasta
01_calibration_seed42_tree.nwk
    A neutral codon alignment (omega = 1.0 everywhere, seed = 42).
    Tree: ((A:0.08,B:0.08)AB:0.04,(C:0.08,D:0.08)CD:0.04);
    n_sites = 1000 codons, 4 taxa.

    Running run_phylospect() on this alignment should yield NOE p-values
    that are NOT significant (p > 0.05 on all branches), consistent with
    the null calibration result in Experiment 01 (KS p = 0.815,
    mean p = 0.494, FPR = 0.058 across 30 replicates).

02_power_omega3_n1000_seed0.fasta
02_power_omega3_n1000_seed0_tree.nwk
    An alignment with episodic positive selection on branch A
    (omega_foreground = 3.0, omega_background = 1.0, seed = 0).
    Same tree and n_sites as above.

    Running run_phylospect() on this alignment should yield a significant
    NOE p-value on branch A (expected p < 0.05 approximately 50% of the
    time at these parameters, per Experiment 02 power results).

HOW TO REPRODUCE THESE FILES
------------------------------
From the repository root:

    python scripts/generate_archived_simulations.py

The script uses the same simulate_episodic_alignment_numpy() function and
fixed seeds as the main experiment scripts; any Python environment with
numpy >= 1.24 will reproduce byte-identical outputs.

HOW TO VERIFY CALIBRATION USING THE ARCHIVED ALIGNMENT
--------------------------------------------------------
    python - << 'EOF'
    import sys; sys.path.insert(0, '.')
    import numpy as np
    from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies
    from phylospect.pipeline import run_phylospect

    def read_fasta(path):
        aln = {}; name = None; chunks = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if name: aln[name] = ''.join(chunks)
                    name = line[1:].split()[0]; chunks = []
                else: chunks.append(line)
            if name: aln[name] = ''.join(chunks)
        return aln

    aln = read_fasta('data/simulated/01_calibration_seed42.fasta')
    tree = open('data/simulated/01_calibration_seed42_tree.nwk').read().strip()
    pi = uniform_codon_frequencies()
    Q0 = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
    results = run_phylospect(tree, aln, Q0=Q0, n_perm=100, rng=np.random.default_rng(0))
    for branch, r in results.items():
        print(f"  {branch}: NOE={r['NOE']:.4f}  p={r['pvalue']:.3f}")
    EOF

Expected output: p-values > 0.05 on all branches (all non-significant under neutrality).

FULL SIMULATION ARCHIVE
-----------------------
The complete simulation outputs (all 30 replicates × 9 experiments, ~3 GB)
are deposited at: [Zenodo DOI — to be added at acceptance]

Regenerating the full archive:
    python experiments/01_calibration.py   # ~10 min
    python experiments/02_power.py         # ~6 h
    python experiments/03_alignment_length.py
    python experiments/04_paml_comparison.py  # requires codeml
    python experiments/05_runtime.py          # requires codeml
    python experiments/06_branch_length.py
    python experiments/07_basis_sensitivity.py
    python experiments/08_misspecification.py  # requires codeml
    python experiments/09_empirical_v4.py data/empirical/lysozyme.fasta \\
        data/empirical/lysozyme.tre lysozyme data/empirical/lysozyme_clades.txt
    python experiments/09_empirical_v4.py data/empirical/cd2.fasta \\
        data/empirical/cd2.tre cd2 data/empirical/cd2_clades.txt
"""

readme_path = OUT_DIR / "README.txt"
readme_path.write_text(readme_text)
print(f"\n  Written: {readme_path}")

print("\n" + "=" * 60)
print("DONE. Files written to data/simulated/:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
print("=" * 60)
print("\nNow commit these files:")
print("  git add data/simulated/")
print("  git commit -m 'Add archived representative simulation outputs (R1.4)'")
print("  git push")
