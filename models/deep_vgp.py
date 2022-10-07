import gpflow as gpf
import numpy as np
from gpflow import default_float, default_jitter
from gpflow.base import RegressionData, InputData, MeanAndVariance
from gpflow.conditionals import conditional
from gpflow.kernels import Kernel
from gpflow.kullback_leiblers import gauss_kl
from gpflow.likelihoods import Likelihood
import tensorflow as tf


class DeepVGP(gpf.models.VGP):
    def __init__(self, data: RegressionData, kernel: Kernel, likelihood: Likelihood, num_outputs):
        super().__init__(data, kernel, likelihood, num_latent_gps=1)

        X, Y = self.data
        self.num_dim = X.shape[1] - 2  # number of columns minus 2 for label columns
        self.num_outputs = num_outputs  # number of outputs
        self.num_layers = int(tf.reduce_max(X[:, -1])) + 1

        tf.config.run_functions_eagerly(True)  # currently there's a bug where this needs to be enabled for gather_nd
        # This turns X and Y into ((num_examples x num_outputs) x 2) separated by labels
        self.Xs = [X[:, :-1]]
        self.Ys = [Y[:, :-1]]
        for i in range(self.num_layers - 1):
            self.Xs.append(tf.gather_nd(indices=tf.where(X[:, -1] > i), params=X[:, :-1]))
            self.Ys.append(tf.gather_nd(indices=tf.where(Y[:, -1] > i), params=Y[:, :-1]))

        tf.config.run_functions_eagerly(False)  # turn this back off for performance

    def elbo(self) -> tf.Tensor:
        # Get prior KL.
        KL = gauss_kl(self.q_mu, self.q_sqrt)
        L = 0

        for i in range(self.num_layers):
            # Get conditionals
            K = self.kernel(self.Xs[i]) + tf.eye(self.Xs[i].shape[0], dtype=default_float()) * default_jitter()
            L = tf.linalg.cholesky(K)
            fmean = tf.linalg.matmul(L, self.q_mu) + self.mean_function(self.Xs[i])  # [NN, ND] -> ND
            q_sqrt_dnn = tf.linalg.band_part(self.q_sqrt, -1, 0)  # [D, N, N]
            L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent_gps, 1, 1]))
            LTA = tf.linalg.matmul(L_tiled, q_sqrt_dnn)  # [D, N, N]
            fvar = tf.reduce_sum(tf.square(LTA), 2)

            fvar = tf.transpose(fvar)

            # Get variational expectations.
            var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Ys[i])
            L += tf.reduce_sum(var_exp)

        return L - tf.convert_to_tensor(self.num_layers) * KL
