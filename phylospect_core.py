import numpy as np

def compute_deltaQ(Qb, Qpool):
    return Qb - Qpool

def signed_ICE(deltaQ):
    return deltaQ.mean()

def ICE_norm(deltaQ):
    return np.linalg.norm(deltaQ, ord='fro')

def projection_stat(deltaQ, B):
    return np.linalg.norm(B.T @ deltaQ.flatten())
