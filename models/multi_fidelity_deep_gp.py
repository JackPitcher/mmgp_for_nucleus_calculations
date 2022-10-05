import copy
from typing import List, Tuple, Union

import gpflow as gpf
import tensorflow as tf
import numpy as np
from gpflow.base import TensorLike

from kernels.neural_network_kernel import NeuralNetKernel

from .model_trainer import ModelTrainer


class MultiFidelityDeepGPTrainer(ModelTrainer):
    VALID_KERNEL_NAMES = ['RBF', 'NeuralNetwork']

    def __init__(self, data: TensorLike, num_outputs: int, optimizer_name: str):
        """
        Instantiates a multi fidelity deep GP trainer.
        :param data: The data, in the form (X, Y). Note that X and Y should have n_dim + 2 columns:
                        1. The n_dim columns are the actual data in n dimensions.
                        2. The second column should be labels for which output is which, i.e.
                           which function is evaluated.
                        3. The last column should be labels for the fidelity. Highest fidelity has
                           the highest number, lowest fidelity should be 0.
                    Hence, the data should have (num_examples x num_outputs) rows and n_dim + 2 columns.
        :param num_outputs: The number of outputs
        :param optimizer_name: The name of the optimizer
        """
        super().__init__(data=data,
                         optimizer_name=optimizer_name,
                         num_outputs=num_outputs)
        X, Y = self.data

        tf.config.run_functions_eagerly(True)  # currently there's a bug where this needs to be enabled for gather_nd
        # This turns X and Y into ((num_examples x num_outputs) x 2) separated by labels
        self.Xs = [X[:, :-1]]
        self.Ys = [Y[:, :-1]]
        for i in range(self.num_layers - 1):
            self.Xs.append(tf.gather_nd(indices=tf.where(X[:, -1] > i), params=X[:, :-1]))
            self.Ys.append(tf.gather_nd(indices=tf.where(Y[:, -1] > i), params=Y[:, :-1]))

        tf.config.run_functions_eagerly(False)  # turn this back off for performance

        self.trained_models = []
        self.kernels = []
        self.likelihoods = []
        self.model_names = []

    def construct_model(self, model_names, base_kernel, likelihood_name):
        """
        Constructs a list of kernels and likelihoods to use for the models.
        :param model_names: A list of names for the models; i.e. which models to use
        :param base_kernel: The base kernel to use; i.e. RBF
        :param likelihood_name: The likelihood to use; i.e. Gaussian
        :return: None
        """
        assert len(model_names) == self.num_layers

        deep_kernels = self.get_deep_kernel(base_kernel)
        self.kernels = [self.get_kernel(base_kernel)] + deep_kernels

        likelihood = gpf.likelihoods.SwitchedLikelihood(
            [self.get_likelihood(likelihood_name) for _ in range(self.num_outputs)])
        self.likelihoods = [copy.deepcopy(likelihood) for _ in range(self.num_layers)]

        self.model_names = model_names

    def train_deep_model(self):
        """
        Trains a deep Gaussian Process model by iterating over the layers.
        :return: None
        """
        for i in range(self.num_layers):
            model = self.get_model(model_name=self.model_names[i],
                                   kernel=self.kernels[i],
                                   likelihood=self.likelihoods[i],
                                   data=(self.Xs[i], self.Ys[i]))
            print(f"Training Model {i + 1}...")
            self.train_model(model=model)
            print(f"Done training model {i + 1}!")
            self.trained_models.append(model)
            if i < self.num_layers - 1:
                mu, _ = model.predict_f(self.Xs[i + 1])
                self.Xs[i + 1] = tf.concat((self.Xs[i + 1], mu), axis=1)

    def predict(self, X_test):
        """
        Predicts the mean and variance of a deep model.
        :param X_test: The test points
        :return: The mean and variance of the highest fidelity
        """
        mu = tf.zeros(X_test.shape)
        var = tf.zeros((X_test.shape[0], X_test.shape[0]))
        for i in range(self.num_layers):
            mu, var = self.trained_models[i].predict_f(X_test)
            X_test = tf.concat((X_test[:, :-1], mu), axis=1) if i > 0 else tf.concat((X_test, mu), axis=1)
        return mu, var

    def get_deep_kernel(self, kernel_name: str) -> List[gpf.kernels.Kernel]:
        """
        Gets a kernel for any GP deeper than the first layer.
        :param kernel_name: The name of the kernel.
        :return: A list of kernels, one for each layer past the first
        """
        kernels = []
        for i in range(self.num_layers - 1):
            coreg = gpf.kernels.Coregion(output_dim=self.num_outputs, rank=self.num_dim, active_dims=[self.num_dim])
            coreg.W = np.random.rand(self.num_outputs, self.num_dim)
            if kernel_name == 'RBF':
                kernels.append(
                    (gpf.kernels.RBF(active_dims=[self.num_dim + 1]) * gpf.kernels.RBF(active_dims=list(range(self.num_layers - 1)))
                     + gpf.kernels.RBF(active_dims=list(range(self.num_layers - 1)))) * coreg)
            elif kernel_name == 'NeuralNetwork':
                kernels.append((NeuralNetKernel(base_kernel=gpf.kernels.RBF(), active_dims=[self.num_dim + 1])
                                * NeuralNetKernel(base_kernel=gpf.kernels.RBF(), active_dims=list(range(self.num_layers - 1)))
                                + NeuralNetKernel(base_kernel=gpf.kernels.RBF(),
                                                  active_dims=list(range(self.num_layers - 1))))
                               * coreg)
            else:
                raise Exception(f"Please make sure all kernels are valid. "
                                f"{kernel_name} is not one of {self.VALID_KERNEL_NAMES}.")
        return kernels
