import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Cholesky(gpf.kernels.Kernel):
    def __init__(self):
        super().__init__()

    def K(self, X1, X2=None):
        L = tfp.math.fill_triangular(X1, upper=False)
        L_transpose = tf.linalg.matrix_transpose(L)
        K = tf.linalg.matmul(L, L_transpose)
        if X2 is None:
            return K
        return K + tf.math.multiply(tf.eye(X1.shape[0]), X2)

    def K_diag(self, X):
        return tf.linalg.diag(self.K(X))