import numpy as np

def within_site_permutation_null(deltaQ, B, n_perm=200):
    """
    Generate within-site permutation null distribution for Mb.

    Parameters
    ----------
    deltaQ : ndarray
        Operator difference matrix (61x61).
    B : ndarray
        Spectral basis (61^2 x k).
    n_perm : int
        Number of permutations.

    Returns
    -------
    null_stats : ndarray
        Null distribution of projection statistics.
    """

    codons = deltaQ.shape[0]
    dim = codons * codons

    null_stats = np.zeros(n_perm)

    for i in range(n_perm):
        perm = deltaQ.copy()

        # permute rows independently = within-site permutation
        for r in range(codons):
            perm[r, :] = np.random.permutation(perm[r, :])

        vec = perm.reshape(dim)
        null_stats[i] = np.linalg.norm(B.T @ vec)

    return null_stats
