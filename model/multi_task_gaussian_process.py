"""Uses coregionalization to construct a multi-output Gaussian process for multiple tasks"""
import gpflow as gpf
import numpy as np
import tensorflow as tf

from typing import Any

from gpflow.base import InputData, MeanAndVariance

gpf.config.set_default_float(np.float64)


class MultiTaskGaussianProcess(gpf.models.BayesianModel, gpf.models.InternalDataTrainingLossMixin):
    """Subclasses GPFlow models to make use of the GPFlow optimization strategies"""

    MAXITER = 1000

    def __init__(self, X, Y, name=None):
        """
        Instantiates the model with the data.
        :param X: The training data
        :param Y: The data evaluated at a function
        :param name: The name of the model
        """
        super().__init__(name=name)

        # Reshape data if necessary: all data should be 2D arrays to work with the kernels
        if len(X.shape) == 1:
            X = np.reshape(X, (X.shape[0], 1))
        if len(Y.shape) == 1:
            Y = np.reshape(Y, (Y.shape[0], 1))

        # Save data to the model
        self.X = X.copy()
        self.Y = Y.copy()

        self.num_data, self.num_dims = X.shape
        _, self.num_outputs = Y.shape

        # Define any parameters not already in the kernels.
        self.a = np.random.rand(self.num_outputs, self.num_dims)  # is this the best way to do this?

        # Define kernels
        self.input_kernel = gpf.kernels.SquaredExponential()
        self.output_kernel = gpf.kernels.Coregion(output_dim=self.num_outputs, rank=self.num_dims)
        self.output_kernel.W = np.random.rand(self.num_outputs, self.num_dims)
        self.noise_kernel = gpf.kernels.White()

    def maximum_log_likelihood_objective(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        input_corr = self.input_kernel.K(self.X)
        output_corr = self.output_kernel.K(self.a)
        noise = self.kronecker(self.noise_kernel.K(self.X))
        combined_corr = self.kronecker(input_corr, output_corr)
        total_corr = noise + combined_corr

        flat_y = tf.reshape(self.Y, shape=(self.num_data * self.num_outputs, 1))
        inner_product = tf.transpose(flat_y) @ tf.linalg.inv(total_corr) @ flat_y

        log_det = tf.linalg.logdet(total_corr)

        return tf.reduce_sum(-0.5 * (inner_product + log_det + self.num_data * tf.cast(tf.math.log(2 * np.pi),
                                                                                       dtype=tf.float64)))

    def kronecker(self, mat1, mat2=None):
        input_operator = tf.linalg.LinearOperatorFullMatrix(mat1)
        if mat2 is not None:
            output_operator = tf.linalg.LinearOperatorFullMatrix(mat2)
        else:
            output_operator = tf.linalg.LinearOperatorFullMatrix(tf.eye(self.num_outputs, dtype=tf.float64))
        return tf.linalg.LinearOperatorKronecker([input_operator, output_operator]).to_dense()

    def predict_f(self, Xnew: InputData) -> MeanAndVariance:
        if len(Xnew.shape) == 1:
            Xnew = np.reshape(Xnew, (Xnew.shape[0], 1))
        test_input_corr = self.input_kernel.K(Xnew, self.X)
        input_corr = self.input_kernel.K(self.X)
        output_corr = self.output_kernel.K(self.a)
        noise = self.kronecker(self.noise_kernel(self.X))
        combined_corr = self.kronecker(input_corr, output_corr)
        test_combined_corr = self.kronecker(test_input_corr, output_corr)
        total_old_corr = noise + combined_corr

        flat_y = tf.reshape(self.Y, shape=(self.num_data * self.num_outputs, 1))
        mean = test_combined_corr @ tf.linalg.inv(total_old_corr) @ flat_y

        test_test_corr = self.kronecker(self.input_kernel.K(Xnew), output_corr)

        variance = test_test_corr - test_combined_corr @ tf.linalg.inv(total_old_corr) @ tf.transpose(test_combined_corr)
        return mean, variance


