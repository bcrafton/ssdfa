import numpy as np


def whiten1(X, method='zca'):
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    U, Lambda, _ = np.linalg.svd(Sigma)
    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
    return np.dot(X_centered, W.T)

def whiten2(X, method='zca'):    
    shape = np.shape(X)
    X = np.reshape(X, (shape[0], -1))
    
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, ddof=1, keepdims=True)
    scale = std + 1.
    X = X - mean
    X = X / scale

    cov = np.cov(X.T, ddof=1)
    mean = np.mean(X, axis=0, keepdims=True)
    [D, V] = np.linalg.eig(cov)

    a = np.diag(np.sqrt(1./(D + 0.1)))
    b = np.dot(V, a)
    c = np.dot(b, V.T)
    P = c
    X = np.dot(X - mean, P)
    
    X = np.reshape(X, shape)
    
    return X
