import gpflow as gpf
import numpy as np
import tensorflow as tf
import copy
import tensorflow_probability as tfp

from models.multi_task_gp import MultiTaskGP


class DiffModel(gpf.models.GPModel, gpf.models.InternalDataTrainingLossMixin):
    """TODO: This is not working properly. When training, Cholesky decomposition gives NaN values. Debug!!!"""

    def __init__(self, data, kernel: gpf.kernels.Kernel, likelihood: gpf.likelihoods.Likelihood, num_outputs):
        super().__init__(kernel, likelihood, num_latent_gps=1)
        X, Y = data

        tf.config.run_functions_eagerly(True)  # currently there's a bug where this must be true for tf.gather_nd

        # This turns X and Y into ((num_examples x num_outputs) x 2) separated by labels
        self.X_lo_fi = tf.gather_nd(indices=tf.where(X[:, -1] == 0), params=X[:, :-1])
        self.Y_lo_fi = tf.gather_nd(indices=tf.where(Y[:, -1] == 0), params=Y[:, :-1])
        self.X_hi_fi = tf.gather_nd(indices=tf.where(X[:, -1] == 1), params=X[:, :-1])
        self.Y_hi_fi = tf.gather_nd(indices=tf.where(Y[:, -1] == 1), params=Y[:, :-1])

        self.num_lf_data = int(self.X_lo_fi.shape[0] / num_outputs)
        self.num_hf_data = int(self.X_hi_fi.shape[0] / num_outputs)

        # sample equally and randomly from the three outputs
        indices = np.random.choice(range(self.num_lf_data), size=self.num_hf_data, replace=False)
        self.X_lf_trunc = tf.concat([tf.gather_nd(indices=np.reshape(indices, (self.num_hf_data, 1)),
                                                  params=self.X_lo_fi[i:i + self.num_lf_data, :]) for i in range(3)],
                                    axis=0)
        self.Y_lf_trunc = tf.concat([tf.gather_nd(indices=np.reshape(indices, (self.num_hf_data, 1)),
                                                  params=self.Y_lo_fi[i:i + self.num_lf_data, :]) for i in range(3)],
                                    axis=0)

        tf.config.run_functions_eagerly(False)  # change back to false for faster training

        self.model = MultiTaskGP(
            data=(self.X_lo_fi, self.Y_lo_fi),
            kernel=kernel,
            likelihood=copy.deepcopy(likelihood),
            num_outputs=3
        )

        gpf.optimizers.Scipy().minimize(self.model.training_loss,
                                        variables=self.model.trainable_variables,
                                        method="L-BFGS-B")
        print("Done training Multi Task Model!")

        self.rho = gpf.Parameter(tf.ones((int(self.num_hf_data * num_outputs), 1)), transform=tfp.bijectors.Exp())

    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        Ktt = self.kernel(Xnew)
        Ktn = self.kernel(Xnew, self.X_hi_fi)
        Knn = self.kernel(self.X_hi_fi)

        var = self.get_variance(self.num_hf_data)

        Knnvar = Knn + var

        mean = Ktn @ tf.linalg.inv(Knnvar) @ tf.expand_dims(self.Y_hi_fi[:, 0], axis=1)

        variance = Ktt - Ktn @ tf.linalg.inv(Knnvar) @ tf.transpose(Ktn)

        return mean, variance

    def get_variance(self, size: float, likelihood: gpf.likelihoods.Likelihood = None) -> tf.Tensor:
        if likelihood is None:
            likelihood = self.likelihood
        variances = [lik.variance for lik in likelihood.likelihoods]
        n = int(size)
        return tf.linalg.diag(tf.concat(
            [tf.linalg.diag_part(tf.fill((n, n), variance)) for variance in variances],
            axis=0))

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        K = self.kernel(self.X_hi_fi)
        Y_hi_fi = tf.expand_dims(self.Y_hi_fi[:, 0], axis=1)
        Y_lo_fi = tf.expand_dims(self.Y_lf_trunc[:, 0], axis=1)
        d = Y_hi_fi + tf.math.multiply(self.rho, Y_lo_fi)
        hi_fi_var = self.get_variance(self.num_hf_data)
        var = hi_fi_var + tf.math.multiply(self.rho, self.get_variance(size=self.num_hf_data,
                                                                       likelihood=self.model.likelihood))

        Kvar = K + var
        d_Kvar_d = tf.linalg.matrix_transpose(d) @ tf.linalg.inv(Kvar) @ d
        log_det = tf.linalg.logdet(Kvar)

        return -tf.reduce_sum(0.5 * (
                d_Kvar_d + log_det + tf.cast(self.num_hf_data * tf.math.log(2 * np.pi), dtype=tf.float64)))
