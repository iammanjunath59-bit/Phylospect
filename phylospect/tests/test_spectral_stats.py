"""
tests/test_spectral_stats.py
============================
Unit tests for phylospect.statistics and the spectral basis builder.

Run from the repo root:
    python tests/test_spectral_stats.py

No external dependencies beyond numpy (already required by the package).
"""

import sys, pathlib, traceback
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phylospect.gy94 import build_gy94_generator, uniform_codon_frequencies

# ── minimal test framework (no pytest needed) ─────────────────────────────────

_passed = 0
_failed = 0
_errors = []

def _run(name, fn):
    global _passed, _failed
    try:
        fn()
        print(f"  PASS  {name}")
        _passed += 1
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        {type(e).__name__}: {e}")
        _failed += 1
        _errors.append((name, traceback.format_exc()))

def assert_approx(a, b, tol=1e-10, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(
            f"{msg} Expected {b}, got {a} (diff={abs(a-b):.2e})")

def assert_allclose(A, B, atol=1e-10, msg=""):
    if not np.allclose(A, B, atol=atol):
        raise AssertionError(
            f"{msg} Max diff = {np.abs(np.array(A) - np.array(B)).max():.2e}")

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_dQ(seed=0):
    rng = np.random.default_rng(seed)
    dQ  = rng.standard_normal((61, 61))
    np.fill_diagonal(dQ, 0.0)
    return dQ

def _build_basis(delta_Qs, k):
    vecs = np.array([dq.flatten() for dq in delta_Qs]).T
    U, _, _ = np.linalg.svd(vecs, full_matrices=False)
    return U[:, :k]

def _mb(dQ, basis):
    return float(np.linalg.norm(basis.T @ dQ.flatten()))

def _get_noe():
    from phylospect import statistics as s
    for name in ('signed_ICE', 'NOE', 'noe'):
        if hasattr(s, name):
            return getattr(s, name)
    raise ImportError("No NOE function found in phylospect.statistics "
                      "(tried: signed_ICE, NOE, noe)")

# ── GY94 tests ────────────────────────────────────────────────────────────────

print("\n=== GY94 model ===")

def t_rows_sum_zero():
    pi = uniform_codon_frequencies()
    Q  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
    assert_allclose(Q.sum(axis=1), np.zeros(61), atol=1e-10,
                    msg="Row sums non-zero.")
_run("rows sum to zero", t_rows_sum_zero)

def t_off_diag_nonneg():
    pi = uniform_codon_frequencies()
    Q  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
    mask = ~np.eye(61, dtype=bool)
    if Q[mask].min() < -1e-12:
        raise AssertionError(f"Negative off-diagonal: {Q[mask].min():.2e}")
_run("off-diagonal non-negative", t_off_diag_nonneg)

def t_diag_nonpositive():
    pi = uniform_codon_frequencies()
    Q  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
    if Q.diagonal().max() > 1e-12:
        raise AssertionError(f"Positive diagonal: {Q.diagonal().max():.2e}")
_run("diagonal non-positive", t_diag_nonpositive)

def t_omega_changes_Q():
    pi = uniform_codon_frequencies()
    Q1 = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=False)
    Q3 = build_gy94_generator(omega=3.0, kappa=2.0, pi=pi, scale=False)
    if np.allclose(Q1, Q3):
        raise AssertionError("omega=1 and omega=3 give identical Q")
_run("omega=1 != omega=3", t_omega_changes_Q)

# ── NOE tests ─────────────────────────────────────────────────────────────────

print("\n=== NOE statistic ===")

def t_noe_zero():
    noe = _get_noe()
    v = noe(np.zeros((61, 61)))
    assert_approx(v, 0.0, tol=1e-12, msg="NOE(0):")
_run("zero for zero deviation", t_noe_zero)

def t_noe_positive():
    pi  = uniform_codon_frequencies()
    Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
    Qfg = build_gy94_generator(omega=5.0, kappa=2.0, pi=pi, scale=True)
    v   = _get_noe()(Qfg - Q0)
    if v <= 0:
        raise AssertionError(f"Expected positive NOE, got {v}")
_run("positive for elevated nonsynonymous", t_noe_positive)

def t_noe_negative():
    pi  = uniform_codon_frequencies()
    Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
    Qfg = build_gy94_generator(omega=0.1, kappa=2.0, pi=pi, scale=True)
    v   = _get_noe()(Qfg - Q0)
    if v >= 0:
        raise AssertionError(f"Expected negative NOE, got {v}")
_run("negative for suppressed nonsynonymous", t_noe_negative)

def t_noe_antisymmetric():
    pi  = uniform_codon_frequencies()
    Q0  = build_gy94_generator(omega=1.0, kappa=2.0, pi=pi, scale=True)
    Qfg = build_gy94_generator(omega=3.0, kappa=2.0, pi=pi, scale=True)
    dQ  = Qfg - Q0
    noe = _get_noe()
    assert_approx(noe(-dQ), -noe(dQ), tol=1e-10, msg="NOE(-dQ) != -NOE(dQ):")
_run("sign flips with negated dQ", t_noe_antisymmetric)

def t_noe_scalar():
    noe = _get_noe()
    for seed in range(5):
        r = noe(_make_dQ(seed))
        if not (np.isscalar(r) or np.array(r).ndim == 0):
            raise AssertionError(f"NOE returned non-scalar shape {np.shape(r)}")
_run("returns scalar", t_noe_scalar)

# ── Spectral basis tests ───────────────────────────────────────────────────────

print("\n=== Spectral basis ===")

def _sample_basis(k=10, n=20, seed=42):
    rng = np.random.default_rng(seed)
    return _build_basis([rng.standard_normal((61, 61)) for _ in range(n)], k)

def t_basis_shape():
    for k in [5, 10, 15]:
        B = _sample_basis(k=k)
        if B.shape != (3721, k):
            raise AssertionError(f"k={k}: expected (3721,{k}), got {B.shape}")
_run("correct shape for each k", t_basis_shape)

def t_basis_orthonormal():
    B = _sample_basis(k=10)
    assert_allclose(B.T @ B, np.eye(10), atol=1e-10,
                    msg="B^T B != I:")
_run("columns orthonormal", t_basis_orthonormal)

def t_basis_unit_norms():
    B = _sample_basis(k=15)
    assert_allclose(np.linalg.norm(B, axis=0), np.ones(15), atol=1e-10,
                    msg="Column norms != 1:")
_run("unit norm columns", t_basis_unit_norms)

def t_mb_nonneg():
    B = _sample_basis()
    for seed in range(10):
        v = _mb(_make_dQ(seed), B)
        if v < 0:
            raise AssertionError(f"M_b < 0: {v}")
_run("projection stat non-negative", t_mb_nonneg)

def t_mb_zero():
    B = _sample_basis(k=5)
    v = _mb(np.zeros((61, 61)), B)
    assert_approx(v, 0.0, tol=1e-12, msg="M_b(0):")
_run("projection stat zero for zero dQ", t_mb_zero)

def t_mb_homogeneous():
    B  = _sample_basis(k=5)
    dQ = _make_dQ(seed=7)
    mb = _mb(dQ, B)
    for c in [2.0, 0.5, -3.0]:
        got = _mb(c * dQ, B)
        exp = abs(c) * mb
        assert_approx(got, exp, tol=1e-10,
                      msg=f"Homogeneity failed for c={c}:")
_run("projection homogeneity |c|*M_b", t_mb_homogeneous)

def t_mb_larger_k():
    rng = np.random.default_rng(0)
    dqs = [rng.standard_normal((61, 61)) for _ in range(30)]
    B5  = _build_basis(dqs, k=5)
    B20 = _build_basis(dqs, k=20)
    dQ  = _make_dQ(seed=99)
    m5  = _mb(dQ, B5)
    m20 = _mb(dQ, B20)
    if m20 < m5 - 1e-10:
        raise AssertionError(f"M_b(k=20)={m20:.6f} < M_b(k=5)={m5:.6f}")
_run("larger k gives >= projection", t_mb_larger_k)

# ── summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Results: {_passed} passed, {_failed} failed")
if _errors:
    print("\nFailed tests — full tracebacks:")
    for name, tb in _errors:
        print(f"\n--- {name} ---")
        print(tb)
sys.exit(0 if _failed == 0 else 1)
