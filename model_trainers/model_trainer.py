from typing import List

import gpflow as gpf
import numpy as np
import tensorflow as tf

from models.deep_vgp import DeepVGP
from models.multi_task_gp import MultiTaskGP
from models.difference_model import DiffModel
from models.multi_fidelity_gp import MultiFidelityGP

from kernels.neural_network_kernel import NeuralNetKernel


class ModelTrainer:
    MAXITER = 10000

    VALID_MODEL_NAMES = ['multi-task-gp', 'VGP', 'GPR', 'GPMC', 'difference', 'DeepVGP']
    VALID_OPTIMIZERS = ['scipy', 'adam']
    VALID_KERNEL_NAMES = ['RBF', 'Matern12', 'Matern32', 'Matern52', 'ArcCosine', 'Coregion', 'White', 'NeuralNetwork']
    VALID_LIKELIHOOD_NAMES = ['Gaussian', 'Exponential', 'StudentT', 'Gamma', 'Beta']

    def __init__(self, data, optimizer_name, num_outputs):
        X, Y = data
        self.num_dim = X.shape[1] - 2  # number of columns minus 2 for label columns
        self.num_outputs = num_outputs  # number of outputs
        self.data = data
        self.num_layers = int(tf.reduce_max(X[:, -1])) + 1

        self.optimizer = self.get_optimizer(optimizer_name)

        self.model = None

    def construct_model(self, model_names, base_kernel, likelihood_name):
        kernel = self.get_multioutput_kernel(base_kernel)
        likelihood = gpf.likelihoods.SwitchedLikelihood(
            [self.get_likelihood(likelihood_name) for _ in range(self.num_outputs)])

        self.model = self.get_model(model_names,
                                    kernel=kernel,
                                    likelihood=likelihood,
                                    data=self.data)

    def get_likelihood(self, likelihood_name: str) -> gpf.likelihoods.ScalarLikelihood:
        if isinstance(likelihood_name, gpf.likelihoods.ScalarLikelihood):
            return likelihood_name
        if likelihood_name == 'Gaussian':
            return gpf.likelihoods.Gaussian()
        if likelihood_name == 'Exponential':
            return gpf.likelihoods.Exponential()
        if likelihood_name == 'StudentT':
            return gpf.likelihoods.StudentT()
        if likelihood_name == 'Gamma':
            return gpf.likelihoods.Gamma()
        if likelihood_name == 'Beta':
            return gpf.likelihoods.Beta()
        raise Exception(
            f"Please enter a valid likelihood name. {likelihood_name} is not in {self.VALID_LIKELIHOOD_NAMES}.")

    def get_kernel(self, kernel_name: str, active_dims: List[int] = None) -> gpf.kernels.Kernel:
        if isinstance(kernel_name, gpf.kernels.Kernel):
            return kernel_name
        if kernel_name == 'RBF':
            return gpf.kernels.RBF(active_dims=active_dims)
        if kernel_name == 'Matern12':
            return gpf.kernels.Matern12(active_dims=active_dims)
        if kernel_name == 'Matern32':
            return gpf.kernels.Matern32(active_dims=active_dims)
        if kernel_name == 'Matern52':
            return gpf.kernels.Matern52(active_dims=active_dims)
        if kernel_name == 'ArcCosine':
            return gpf.kernels.ArcCosine(active_dims=active_dims)
        if kernel_name == 'Linear':
            return gpf.kernels.Linear(active_dims=active_dims)
        if kernel_name == 'NeuralNetwork':
            return NeuralNetKernel(base_kernel=gpf.kernels.RBF(), num_dims=self.num_dim,
                                   active_dims=list(range(self.num_layers - 1)))
        if kernel_name == 'White':
            return gpf.kernels.White()
        raise Exception(
            f"Please make sure all kernels are valid. {kernel_name} is not one of {self.VALID_KERNEL_NAMES}.")

    def get_multioutput_kernel(self, kernel_name: str) -> gpf.kernels.Kernel:
        coreg = gpf.kernels.Coregion(output_dim=self.num_outputs, rank=self.num_dim, active_dims=[self.num_dim])
        coreg.W = np.random.rand(self.num_outputs, self.num_dim)
        active_dims = list(range(self.num_dim - 1))
        return self.get_kernel(kernel_name, active_dims=active_dims) * coreg

    def get_model(self, model_name, kernel, likelihood, data) -> gpf.models.GPModel:
        if isinstance(model_name, gpf.models.GPModel):
            return model_name
        if model_name == 'multi-fidelity-gp':
            return MultiFidelityGP(data, kernel=kernel, likelihood=likelihood, num_outputs=3)
        if model_name == 'multi-task-gp':
            return MultiTaskGP(data, kernel=kernel, likelihood=likelihood, num_outputs=self.num_outputs)
        if model_name == 'VGP':
            return gpf.models.VGP(data, kernel=kernel, likelihood=likelihood,
                                  num_latent_gps=1)
        if model_name == 'DeepVGP':
            return DeepVGP(data, kernel=kernel, likelihood=likelihood, num_outputs=3)
        if model_name == 'GPR':
            return gpf.models.GPR(data, kernel=kernel)
        if model_name == 'GPMC':
            return gpf.models.GPMC(data, kernel=kernel, likelihood=likelihood, num_latent_gps=1)
        if model_name == 'difference':
            return DiffModel(data, kernel=kernel, likelihood=likelihood, num_outputs=self.num_outputs)
        raise Exception(f"Please enter a valid models name: one of {self.VALID_MODEL_NAMES}.")

    def get_optimizer(self, optimizer_name):
        if optimizer_name == 'scipy':
            return gpf.optimizers.Scipy()
        if optimizer_name == 'adam':
            return tf.optimizers.Adam()
        raise Exception(f"Please enter a valid optimizer name: one of {self.VALID_OPTIMIZERS}")

    def train_model(self, model=None, disp=False):
        if model is None:
            model = self.model
        if isinstance(self.optimizer, gpf.optimizers.Scipy):
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for information
            self.optimizer.minimize(model.training_loss,
                                    variables=model.trainable_variables,
                                    options=dict(disp=disp),
                                    method="L-BFGS-B")
            tf.print("Done training model!")
        else:
            tf_optimization_step = tf.function(self.optimization_step)
            for epoch in range(self.MAXITER):
                loss = tf_optimization_step()
                epoch_id = epoch + 1
                if epoch_id % (self.MAXITER // 100) == 0:
                    tf.print(f"Epoch {epoch_id}: Loss (train) {loss}")
            tf.print("Done training model!")

    def optimization_step(self):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.model.training_loss()
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def predict(self, X_test):
        mean, var = self.model.predict_f(X_test)
        return mean, var
