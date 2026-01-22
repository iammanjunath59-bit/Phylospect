import numpy as np
from scipy.linalg import qr

def learn_spectral_basis(Q0, omega=1.2, frac=0.3, n_samples=50, n_basis=15):
    """
    Learn a low-dimensional spectral basis from biologically plausible
    operator perturbations.

    Parameters
    ----------
    Q0 : ndarray
        Neutral codon generator (61x61).
    omega : float
        Weak selection inflation used for perturbations.
    frac : float
        Fraction of entries perturbed.
    n_samples : int
        Number of perturbation samples.
    n_basis : int
        Number of basis vectors retained.

    Returns
    -------
    B : ndarray
        Orthonormal spectral basis (61^2 x n_basis).
    """

    codons = Q0.shape[0]
    dim = codons * codons

    X = []

    for _ in range(n_samples):
        Qp = Q0.copy()
        mask = np.random.rand(codons, codons) < frac
        Qp[mask] *= omega
        delta = (Qp - Q0).reshape(dim)
        X.append(delta)

    X = np.vstack(X).T  # shape: (61^2, n_samples)

    Q, _ = qr(X, mode="economic")

    return Q[:, :n_basis]
