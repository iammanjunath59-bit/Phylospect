"""
Experiment 09 v4 — Internal branches only.

Changes from v3:
  - Tests ONLY the internal clade branches in the clade-spec file. Terminal
    branches on small datasets (e.g. lysozyme) consistently give LRT=0
    because the selection signal is on the ancestral lineage, not the tips.
  - Reduced multi-start to 2 ω (1.5, 4.0) since v3 showed alt converged
    consistently across starts; further variation isn't informative.
  - Required clade-spec file (no terminal fallback).

Usage:
    python experiments/09_empirical_v4.py <alignment.fasta> <tree.nwk> <label> <clade_spec.txt>

Runtime:
    PHYLOSPECT: ~15s
    PAML: ~3-5 min per clade × 2 starts × N_clades (lysozyme: ~20-30 min total)
"""

from __future__ import annotations
import sys, os, re, subprocess, tempfile, time, pathlib, shutil
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies, sense_codons
from phylospect.simulate import parse_newick
from phylospect.pipeline import run_phylospect

CODEML_PATH    = r"C:\Users\iamma\phylospect\phylospect\bin\codeml.exe"
N_PERM         = 100
N_SM           = 30
N_SM_NULL      = 12
ALPHA          = 0.05
SEED           = 0
START_OMEGAS   = [1.5, 4.0]   # 2 starts: enough for robustness, fast


# ─── FASTA / validation ────────────────────────────────────────────────

def read_fasta(path):
    out = {}; name = None; chunks = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if name: out[name] = "".join(chunks).upper().replace("-", "")
                name = line[1:].split()[0]; chunks = []
            else:
                chunks.append(line)
        if name: out[name] = "".join(chunks).upper().replace("-", "")
    return out


def validate_alignment(aln):
    if not aln: return False, "Empty"
    L = len(next(iter(aln.values())))
    if L % 3 != 0: return False, f"Length {L} not multiple of 3"
    sense = set(sense_codons())
    for n, s in aln.items():
        if len(s) != L: return False, f"Length mismatch in {n}"
        for k in range(0, L, 3):
            if s[k:k+3] not in sense:
                return False, f"Bad codon '{s[k:k+3]}' in {n} at pos {k}"
    return True, f"OK: {len(aln)} seqs, {L // 3} codons"


def sanitise(name): return re.sub(r"[^A-Za-z0-9_]", "_", name)


def sanitise_alignment_and_tree(aln, tree_str):
    name_map = {}; new_aln = {}
    for orig, seq in aln.items():
        clean = sanitise(orig); base = clean; suffix = 1
        while clean in new_aln:
            suffix += 1; clean = f"{base}_{suffix}"
        new_aln[clean] = seq; name_map[clean] = orig
    new_tree = tree_str
    for clean, orig in sorted(name_map.items(), key=lambda kv: -len(kv[1])):
        if clean != orig:
            new_tree = re.sub(
                r"(?<![A-Za-z0-9_&\.\-])" + re.escape(orig) + r"(?=[:,)\s])",
                clean, new_tree,
            )
    return new_aln, new_tree, name_map


def leaves_under(node):
    if node.is_leaf: return {node.name}
    s = set()
    for c in node.children or []: s |= leaves_under(c)
    return s


def find_mrca(tree, target_leaves):
    target = set(target_leaves)
    for node in tree.iter_postorder():
        if not node.is_leaf and leaves_under(node) == target:
            return node
    best = None; best_size = float("inf")
    for node in tree.iter_postorder():
        if not node.is_leaf:
            ls = leaves_under(node)
            if target.issubset(ls) and len(ls) < best_size:
                best = node; best_size = len(ls)
    return best


def parse_clade_spec(path, name_map_inv):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            if ":" not in line: continue
            name, leaves_str = line.split(":", 1)
            name = name.strip()
            raw = [s.strip() for s in leaves_str.split(",") if s.strip()]
            sanitised = [name_map_inv.get(r, sanitise(r)) for r in raw]
            out[name] = sanitised
    return out


# ─── PAML ──────────────────────────────────────────────────────────────

def make_phylip(aln, n_codons):
    lines = [f" {len(aln)}  {3 * n_codons}"]
    for name, seq in aln.items():
        lines.append(f"{name:<32}{seq}")
    return "\n".join(lines) + "\n"


def write_paml_tree_internal(tree_str, mrca_name):
    tree = parse_newick(tree_str)
    def emit(node):
        if node.is_leaf:
            return f"{node.name}:{node.branch_length}"
        children = ",".join(emit(c) for c in node.children)
        if node.name == mrca_name:
            return f"({children}):{node.branch_length} #1"
        return f"({children}):{node.branch_length}"
    body = emit(tree)
    body = re.sub(r":[\d\.]+\s*$", "", body)
    leaves = [n.name for n in tree.leaves()]
    return f" {len(leaves)}  1\n{body};\n"


def make_ctl(seqfile, treefile, outfile, fix_omega, omega):
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


def parse_lnl(text):
    m = re.search(r"lnL\(ntime:.*?\):\s*(-?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def parse_w_classes(text):
    m = re.search(
        r"site class.*?proportion[\s\d\.]+background w[\s\d\.]+foreground w\s+([\d\.\s]+)",
        text, re.DOTALL,
    )
    return m.group(1).strip() if m else None


def run_paml(aln, paml_tree_str, n_codons, codeml):
    from scipy.stats import chi2
    info = {"alt_lnl": None, "null_lnl": None, "lrt": None,
             "pvalue": None, "alt_w": None}
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "aln.phy"), "w").write(make_phylip(aln, n_codons))
        open(os.path.join(tmp, "tree.nwk"), "w").write(paml_tree_str)

        alt_results = []
        for w0 in START_OMEGAS:
            tag = f"alt_w{w0}"
            ctl = os.path.join(tmp, f"{tag}.ctl")
            out = os.path.join(tmp, f"{tag}.out")
            open(ctl, "w").write(make_ctl("aln.phy", "tree.nwk", f"{tag}.out", 0, w0))
            try:
                subprocess.run([codeml, os.path.basename(ctl)],
                               cwd=tmp, capture_output=True, timeout=600)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            if not os.path.isfile(out): continue
            text = open(out, errors="replace").read()
            lnl = parse_lnl(text)
            if lnl is not None:
                alt_results.append((lnl, parse_w_classes(text)))
        if not alt_results: return None, info

        ctl = os.path.join(tmp, "null.ctl")
        out = os.path.join(tmp, "null.out")
        open(ctl, "w").write(make_ctl("aln.phy", "tree.nwk", "null.out", 1, 1.0))
        try:
            subprocess.run([codeml, "null.ctl"], cwd=tmp,
                           capture_output=True, timeout=600)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None, info
        if not os.path.isfile(out): return None, info
        null_lnl = parse_lnl(open(out, errors="replace").read())
        if null_lnl is None: return None, info

        best_alt = max(alt_results, key=lambda r: r[0])
        info["alt_lnl"]  = best_alt[0]
        info["alt_w"]    = best_alt[1]
        info["null_lnl"] = null_lnl
        info["lrt"]      = max(0.0, 2.0 * (best_alt[0] - null_lnl))
        info["pvalue"]   = 0.5 * float(chi2.sf(info["lrt"], df=1))
        return info["pvalue"], info


def _is_nan_or_none(v):
    if v is None: return True
    try: return bool(np.isnan(v))
    except (TypeError, ValueError): return False


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 5:
        print(__doc__); return 2
    aln_path, tree_path, label, clade_path = sys.argv[1:5]

    print("=" * 70)
    print(f"EXPERIMENT 09 v4 — internal-branch comparison ({label})")
    print("=" * 70)
    codeml_ok = os.path.isfile(CODEML_PATH) or shutil.which(CODEML_PATH)
    print(f"  codeml: {CODEML_PATH if codeml_ok else 'NOT FOUND'}")

    aln_raw = read_fasta(aln_path)
    ok, msg = validate_alignment(aln_raw)
    print(f"  Alignment: {msg}")
    if not ok: return 1
    n_codons = len(next(iter(aln_raw.values()))) // 3

    tree_str = open(tree_path).read().strip()
    if not tree_str.endswith(";"): tree_str += ";"
    aln_clean, tree_clean, name_map = sanitise_alignment_and_tree(aln_raw, tree_str)
    name_map_inv = {v: k for k, v in name_map.items()}

    tree = parse_newick(tree_clean)
    leaf_names = [n.name for n in tree.leaves()]
    if set(leaf_names) != set(aln_clean.keys()):
        print(f"  ERROR: name mismatch")
        print(f"    only tree: {set(leaf_names) - set(aln_clean.keys())}")
        print(f"    only aln : {set(aln_clean.keys()) - set(leaf_names)}")
        return 1
    aln = {n: aln_clean[n] for n in leaf_names}

    clade_specs = parse_clade_spec(clade_path, name_map_inv)
    if not clade_specs:
        print(f"  ERROR: no clades found in {clade_path}")
        return 1

    targets = []
    for cname, cleaves in clade_specs.items():
        m = find_mrca(tree, cleaves)
        if m is None:
            print(f"  WARNING: clade '{cname}' not found in tree, skipping")
            continue
        targets.append((cname, m.name, write_paml_tree_internal(tree_clean, m.name)))
    print(f"  Internal branches to test: {len(targets)}")
    for cname, internal_name, _ in targets:
        print(f"    {cname:<24} -> internal node '{internal_name}'")

    # PHYLOSPECT once.
    # Probe available branches first — short internal branches may have zero
    # sampled substitution events and be absent from Qs, causing KeyError.
    print(f"\n  Probing available branches...", flush=True)
    from phylospect.stochastic_mapping import estimate_branch_Qs_stochastic, _make_P_cache
    from phylospect.simulate import parse_newick as _pn
    pi = uniform_codon_frequencies()
    Q0 = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
    _ptree = _pn(tree_clean)
    _pcache = _make_P_cache(_ptree, Q0)
    _probe_Qs, _ = estimate_branch_Qs_stochastic(
        _ptree, aln, Q0, n_samples=30,
        rng=np.random.default_rng(SEED), P_cache=_pcache,
    )
    available = set(_probe_Qs.keys())
    print(f"    Available Qs keys: {sorted(available)}")

    phy_focal_all = [t[1] for t in targets]
    phy_focal = [b for b in phy_focal_all if b in available]
    skipped = [b for b in phy_focal_all if b not in available]
    if skipped:
        print(f"    WARNING: skipping branches absent from Qs: {skipped}")
        targets = [(cn, ib, pt) for (cn, ib, pt) in targets if ib not in skipped]

    print(f"  Running PHYLOSPECT (n_perm={N_PERM}, n_sm={N_SM})...", flush=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    phy_results = run_phylospect(
        tree_clean, aln, Q0=Q0, n_perm=N_PERM,
        n_sm_samples=N_SM, n_sm_samples_null=N_SM_NULL,
        focal_branches=phy_focal, rng=rng,
    )
    print(f"    PHYLOSPECT: {time.time()-t0:.1f}s for {len(phy_focal)} branches")

    # PAML per clade.
    rows = []
    if codeml_ok:
        print(f"\n  Running PAML (multi-start, internal branches)...", flush=True)
        for (cname, internal, paml_tree) in targets:
            t0 = time.time()
            p, info = run_paml(aln, paml_tree, n_codons, CODEML_PATH)
            dt = time.time() - t0
            p_str = "FAIL" if p is None else f"{p:.4f}"
            print(f"    PAML  {cname:<24} p={p_str}  LRT={info['lrt']}  [{dt:.0f}s]")
            rows.append({
                "clade": cname,
                "internal_node": internal,
                "phylospect_NOE": phy_results[internal]["NOE"],
                "phylospect_pvalue": phy_results[internal]["pvalue"],
                "phylospect_ICEnorm": phy_results[internal]["ICEnorm"],
                "paml_alt_lnL": info["alt_lnl"],
                "paml_null_lnL": info["null_lnl"],
                "paml_LRT": info["lrt"],
                "paml_pvalue": p,
                "paml_alt_w_classes": info["alt_w"],
            })
    else:
        for (cname, internal, _) in targets:
            rows.append({
                "clade": cname, "internal_node": internal,
                "phylospect_NOE": phy_results[internal]["NOE"],
                "phylospect_pvalue": phy_results[internal]["pvalue"],
                "phylospect_ICEnorm": phy_results[internal]["ICEnorm"],
                "paml_alt_lnL": np.nan, "paml_null_lnL": np.nan,
                "paml_LRT": np.nan, "paml_pvalue": np.nan,
                "paml_alt_w_classes": None,
            })

    df = pd.DataFrame(rows)
    out_dir = pathlib.Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)
    csv = out_dir / f"09_empirical_{label}_internal.csv"
    df.to_csv(csv, index=False)
    print(f"\n  Saved: {csv}")

    # Summary.
    print()
    print("=" * 70)
    print(f"SUMMARY — {label} (internal branches)")
    print("=" * 70)
    print(f"  {'clade':<22} {'PHYLOSPECT p':<14} {'PAML p':<10} {'PAML LRT':<10}")
    print("  " + "-" * 60)
    for _, r in df.iterrows():
        phy = r["phylospect_pvalue"]; pal = r["paml_pvalue"]; lrt = r["paml_LRT"]
        phy_s = f"{phy:.4f}" if not _is_nan_or_none(phy) else "NaN"
        pal_s = f"{pal:.4f}" if not _is_nan_or_none(pal) else "NaN"
        lrt_s = f"{lrt:.2f}" if not _is_nan_or_none(lrt) else "NaN"
        print(f"  {r['clade']:<22} {phy_s:<14} {pal_s:<10} {lrt_s:<10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
