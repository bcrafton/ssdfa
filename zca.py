import numpy as np

'''
def whiten(X):
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    U, Lambda, _ = np.linalg.svd(Sigma)
    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
    return np.dot(X_centered, W.T)
'''

def whiten(X):
    N = tf.shape(X)[0]
    N = tf.cast(N, dtype=tf.float32)

    X = tf.reshape(X, (N, -1))
    X_centered = X - tf.reduce_mean(X, axis=0)
    Sigma = tf.matmul(tf.transpose(X_centered), X_centered) / N
    U, Lambda, _ = tf.linalg.svd(Sigma)
    W = np.matmul(U, tf.matmul(tf.linalg.diag(1.0 / tf.sqrt(Lambda + 1e-5)), tf.transpose(U)))
    return np.dot(X_centered, W.T)
