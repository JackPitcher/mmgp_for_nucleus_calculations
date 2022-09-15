import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .cholesky_kernel import Cholesky


class DifferenceModel:

    def __init__(self, hi_fi_x, lo_fi_x, lo_fi_y, hi_fi_y):
        num_tasks = hi_fi_y.shape[0]
        print("Num tasks: ", num_tasks)
        self.x = hi_fi_x
        self.lo_fi_x = lo_fi_x[:self.x.shape[0], :self.x.shape[1]]
        self.lo_fi_y = lo_fi_y[:self.x.shape[0], :self.x.shape[1]]
        self.hi_fi_y = hi_fi_y
        self.rho = tf.Variable(tf.ones(self.x.shape[0]))
        self.a = tf.Variable(tf.ones(int(num_tasks * (num_tasks + 1) / 2)))
        self.cholesky_kernel = Cholesky()
        self.input_kernel = gpf.kernels.SquaredExponential()
        self.white_kernel = gpf.kernels.White()

    def train(self, learning_rate=0.1, num_steps=1000):
        losses = tfp.math.minimize(self.loss,
                                   optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                                   num_steps=num_steps)
        return losses

    def loss(self):
        correlation_mat = self.calculate_covariance_matrix()
        hi_fi_sigma = self.calculate_white_noise_kernel(self.x)
        lo_fi_sigma = self.calculate_white_noise_kernel(self.lo_fi_x)
        total_sigma = self.calculate_difference(hi_fi_sigma, lo_fi_sigma)
        corr_mat_plus_sigma = tf.math.add(correlation_mat, self.kronecker(total_sigma))
        d = self.calculate_difference(self.hi_fi_y[:, 0], self.lo_fi_y[:, 0])
        dK = tf.linalg.matmul(self.kronecker(d), tf.linalg.inv(corr_mat_plus_sigma), transpose_a=True)
        dKd = 0.5 * tf.linalg.matmul(dK, self.kronecker(d))
        log_det = 0.5 * tf.linalg.logdet(corr_mat_plus_sigma)
        num_hi_fi = tf.size(self.x)
        dKd_plus_log_det = tf.math.add(dKd, log_det)
        return tf.reduce_sum(tf.math.add(dKd_plus_log_det, tf.cast(num_hi_fi, float) / tf.cast(tf.constant(2 * np.log(2 * np.pi)), float)))

    def calculate_difference(self, hi_fi_data: tf.Tensor, lo_fi_data: tf.Tensor) -> tf.Tensor:
        return tf.cast(hi_fi_data, dtype=float) - tf.math.multiply(self.rho, tf.cast(lo_fi_data, dtype=float))

    def calculate_covariance_matrix(self) -> tf.Tensor:
        """
        Calculates the covariance matrix over the inputs and outputs
        :param x: Inputs
        :param a: Parameters for the outputs
        :return: The Kronecker product of the input covariance and the output covariance
        """
        input_operator = tf.linalg.LinearOperatorFullMatrix(tf.cast(self.calculate_input_covariance(), dtype=float))
        output_operator = tf.linalg.LinearOperatorFullMatrix(self.calculate_output_covariance())
        return tf.linalg.LinearOperatorKronecker([input_operator, output_operator]).to_dense()

    def kronecker(self, x: tf.Tensor) -> tf.Tensor:
        """
        Calculates a white noise matrix using the parameters sigma
        :return: A TF Tensor that represents the noise of the inputs
        """
        input_operator = tf.linalg.LinearOperatorFullMatrix(tf.linalg.diag(tf.cast(x, dtype=float)))
        output_operator = tf.linalg.LinearOperatorFullMatrix(tf.eye(x.shape[0]))
        return tf.linalg.LinearOperatorKronecker([input_operator, output_operator]).to_dense()

    def calculate_white_noise_kernel(self, x) -> tf.Tensor:
        """
        Calculates a white noise kernel to represent the uncertainty of the inputs
        :return: The white noise kernel matrix
        """
        return self.white_kernel.K_diag(x)

    def calculate_input_covariance(self) -> tf.Tensor:
        """
        Calculates a standard RBF covariance matrix on inputs
        :return: The RBF matrix
        """
        return self.input_kernel.K(self.x)

    def calculate_output_covariance(self) -> tf.Tensor:
        """
        Calculates a kernel using free form parameterization.
        :return: The covariance matrix computed over outputs
        """
        return self.cholesky_kernel.K(self.a)
