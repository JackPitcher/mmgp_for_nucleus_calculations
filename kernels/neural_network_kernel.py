import gpflow as gpf
import tensorflow as tf
from gpflow.base import TensorType
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense

class NeuralNetKernel(gpf.kernels.Kernel):

    def __init__(self, base_kernel, num_dims, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.nn = Sequential(name="neural_network")
        # self.nn.add(Dense(32, activation='relu', dtype=tf.float64))
        # self.nn.add(Dense(16, activation='relu', dtype=tf.float64))
        # self.nn.add(Dense(8, activation='relu', dtype=tf.float64))
        # self.nn.add(Dense(16, activation='relu', dtype=tf.float64))
        # self.nn.add(Dense(32, activation='relu', dtype=tf.float64))
        self.nn.add(Dense(num_dims, activation='relu', dtype=tf.float64))

    def K(self, X: TensorType, X2: TensorType=None) -> tf.Tensor:
        transform_X = self.nn(X)
        transform_X2 = self.nn(X2) if X2 is not None else X2
        return self.base_kernel.K(transform_X, transform_X2)

    def K_diag(self, X: TensorType) -> tf.Tensor:
        transform_X = self.nn(X)
        return self.base_kernel.K(transform_X)
