from typing import Any

import gpflow as gpf
import numpy as np
import tensorflow as tf
from gpflow.models.util import data_input_to_tensor

gpf.config.set_default_float(np.float64)


class MultiTaskGP(gpf.models.GPModel, gpf.models.InternalDataTrainingLossMixin):

    def __init__(self, data, kernel: gpf.kernels.Kernel, likelihood: gpf.likelihoods.Likelihood, num_outputs):
        super().__init__(kernel, likelihood, num_latent_gps=1)
        self.X, self.Y = data
        self.num_outputs = num_outputs
        self.num_examples = self.X.shape[0] / num_outputs

    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        Ktt = self.kernel(Xnew)
        Ktn = self.kernel(Xnew, self.X)
        Knn = self.kernel(self.X)

        var = self.get_variance(self.num_examples)

        Knnvar = Knn + var

        mean = Ktn @ tf.linalg.inv(Knnvar) @ tf.expand_dims(self.Y[:, 0], axis=1)

        variance = Ktt - Ktn @ tf.linalg.inv(Knnvar) @ tf.transpose(Ktn)

        return mean, variance

    def get_variance(self, size) -> tf.Tensor:
        variances = [likelihood.variance for likelihood in self.likelihood.likelihoods]
        n = int(size)
        return tf.linalg.diag(tf.concat(
            [tf.linalg.diag_part(tf.fill((n, n), variance)) for variance in variances],
            axis=0))

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.calculate_loss(self.X, self.Y)

    def calculate_loss(self, X, Y) -> tf.Tensor:
        K = self.kernel(X)
        var = self.get_variance(X.shape[0] / self.num_outputs)
        Kvar = K + var

        d_Kvar_d = tf.expand_dims(Y[:, 0], axis=0) @ tf.linalg.inv(Kvar) @ tf.expand_dims(Y[:, 0], axis=1)
        log_det = tf.linalg.logdet(Kvar)

        return -tf.reduce_sum(0.5 * (
                    d_Kvar_d + log_det + tf.cast(X.shape[0] / self.num_outputs * tf.math.log(2 * np.pi),
                                                 dtype=tf.float64)))

