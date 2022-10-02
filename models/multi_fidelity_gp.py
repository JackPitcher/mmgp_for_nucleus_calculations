import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class MultiFidelityGP(gpf.models.GPModel, gpf.models.InternalDataTrainingLossMixin):

    def __init__(self, data, kernel: gpf.kernels.Kernel, likelihood: gpf.likelihoods.Likelihood,
                 num_outputs):
        """
        Creates a MultiFidelityGP instance.
        :param data: The data, in the form (X, Y). Note that X and Y should have three columns:
                        1. The first column should be the data itself; for X, this should be the
                           data points; for Y, this should be the value of the appropriate function
                           evaluated at the respective points.
                        2. The second column should be labels for which output is which, i.e.
                           which function is evaluated.
                        3. The last column should be labels for the fidelity. Highest fidelity has
                           the highest number, lowest fidelity should be 0.
                    Hence, the data should have (num_examples x num_outputs) rows and 3 columns.
        :param kernel: The kernel to use
        :param likelihood: The likelihood to use
        """
        super().__init__(kernel, likelihood, num_latent_gps=1)
        X, Y = data

        tf.config.run_functions_eagerly(True)  # currently there's a bug where this needs to be enabled for gather_nd
        # This turns X and Y into ((num_examples x num_outputs) x 2) separated by labels
        self.X_lo_fi = tf.gather_nd(indices=tf.where(X[:, -1] == 0), params=X[:, :-1])
        self.Y_lo_fi = tf.gather_nd(indices=tf.where(Y[:, -1] == 0), params=Y[:, :-1])
        self.X_hi_fi = tf.gather_nd(indices=tf.where(X[:, -1] == 1), params=X[:, :-1])
        self.Y_hi_fi = tf.gather_nd(indices=tf.where(Y[:, -1] == 1), params=Y[:, :-1])

        tf.config.run_functions_eagerly(False)  # turn this back off for performance

        self.num_lf_data = int(self.X_lo_fi.shape[0] / num_outputs)
        self.num_hf_data = int(self.X_hi_fi.shape[0] / num_outputs)

        self.num_outputs = num_outputs

        self.data = data

        # Square seems to work best right now
        self.rho = gpf.Parameter(0.2, transform=tfp.bijectors.Square())

    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        """
        Predicts the mean and variance of the models given new X values.
        :param Xnew: The new x values, should be in the form (num_test_examples x num_outputs, 2)
                  where the second column consists of labels for the output.
        :param full_cov: Not implemented
        :param full_output_cov: Not implemented
        :return: The means and variances
        """

        Ktt = self.kernel(Xnew)
        Ktn = self.kernel(Xnew, self.X_hi_fi)
        Knn = self.kernel(self.X_hi_fi)

        var = self.get_variance(self.num_hf_data)

        Knnvar = Knn + var

        mean = Ktn @ tf.linalg.inv(Knnvar) @ tf.expand_dims(self.Y_hi_fi[:, 0], axis=1)

        variance = Ktt - Ktn @ tf.linalg.inv(Knnvar) @ tf.transpose(Ktn)

        return mean, variance

    def get_variance(self, size) -> tf.Tensor:
        """
        Gets the variance of the likelihoods in a diagonal tensor with size (size, size).
        :param size: The size of the tensor for the variance matrix.
        :return: The variance matrix.
        """
        variances = [likelihood.variance for likelihood in self.likelihood.likelihoods]
        n = int(size)
        return tf.linalg.diag(tf.concat(
            [tf.linalg.diag_part(tf.fill((n, n), variance)) for variance in variances],
            axis=0))

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """
        Determines the maximum likelihood objective by calculating the loss.
        :return: The total loss.
        """

        lo_fi_loss = self.calculate_loss(self.X_lo_fi, self.Y_lo_fi)
        hi_fi_loss = self.calculate_loss(self.X_hi_fi, self.Y_hi_fi)

        return self.rho * hi_fi_loss + lo_fi_loss

    def calculate_loss(self, X, Y) -> tf.Tensor:
        """
        Calculates the loss given X and Y.
        :param X: The input data; should be in the form (num_examples x num_outputs, 2)
                  where the second column consists of labels for the output.
        :param Y: The function evaluations; should be in the form (num_examples x num_outputs, 2)
                  where the second column consists of labels for the output.
        :return: A value for the loss
        """
        K = self.kernel(X)
        var = self.get_variance(X.shape[0] / self.num_outputs)
        Kvar = K + var

        d_Kvar_d = tf.expand_dims(Y[:, 0], axis=0) @ tf.linalg.inv(Kvar) @ tf.expand_dims(Y[:, 0], axis=1)
        log_det = tf.linalg.logdet(Kvar)

        return -tf.reduce_sum(0.5 * (
                    d_Kvar_d + log_det + tf.cast(X.shape[0] / self.num_outputs * tf.math.log(2 * np.pi),
                                                 dtype=tf.float64)))
