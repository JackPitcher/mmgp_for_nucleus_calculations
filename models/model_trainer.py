import gpflow as gpf
import numpy as np
import tensorflow as tf

from .multi_task_gp import MultiTaskGP
from .difference_model import DiffModel
from .multi_fidelity_gp import MultiFidelityGP

from kernels.neural_network_kernel import NeuralNetKernel


class ModelTrainer:
    MAXITER = 10000

    VALID_MODEL_NAMES = ['multi-task-gp', 'VGP', 'GPR', 'GPMC', 'difference']
    VALID_OPTIMIZERS = ['scipy']
    VALID_KERNEL_NAMES = ['RBF', 'Matern12', 'Matern32', 'Matern52', 'ArcCosine', 'Coregion', 'White', 'NeuralNetwork']
    VALID_LIKELIHOOD_NAMES = ['Gaussian', 'Exponential', 'StudentT', 'Gamma', 'Beta']

    def __init__(self, model_name, optimizer_name, kernel_names, likelihood_name, X, Y, num_outputs):
        self.num_data, self.num_dim = X.shape
        self.num_outputs = num_outputs  # number of outputs

        self.data = (X, Y)

        self.kernel = gpf.kernels.Product([self.get_kernel(kern, i) for i, kern in enumerate(kernel_names)])
        self.likelihood = gpf.likelihoods.SwitchedLikelihood(
            [self.get_likelihood(likelihood_name) for _ in range(self.num_outputs)])

        self.optimizer = self.get_optimizer(optimizer_name)
        self.model = self.get_model(model_name)

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

    def get_kernel(self, kernel_name: str, i: int) -> gpf.kernels.Kernel:
        if isinstance(kernel_name, gpf.kernels.Kernel):
            return kernel_name
        if kernel_name == 'RBF':
            return gpf.kernels.RBF(active_dims=[0])
        if kernel_name == 'Matern12':
            return gpf.kernels.Matern12(active_dims=[0])
        if kernel_name == 'Matern32':
            return gpf.kernels.Matern32(active_dims=[0])
        if kernel_name == 'Matern52':
            return gpf.kernels.Matern52(active_dims=[0])
        if kernel_name == 'ArcCosine':
            return gpf.kernels.ArcCosine(active_dims=[0])
        if kernel_name == 'NeuralNetwork':
            return NeuralNetKernel(base_kernel=gpf.kernels.RBF(), active_dims=[0])
        if kernel_name == 'Coregion':
            kernel = gpf.kernels.Coregion(output_dim=self.num_outputs, rank=self.num_dim, active_dims=[1])
            kernel.W = np.random.rand(self.num_outputs, self.num_dim)
            return kernel
        if kernel_name == 'White':
            return gpf.kernels.White()
        raise Exception(
            f"Please make sure all kernels are valid. {kernel_name} is not one of {self.VALID_KERNEL_NAMES}.")

    def get_model(self, model_name) -> gpf.models.GPModel:
        if isinstance(model_name, gpf.models.GPModel):
            return model_name
        if model_name == 'multi-fidelity-gp':
            return MultiFidelityGP(self.data, kernel=self.kernel, likelihood=self.likelihood, num_outputs=3)
        if model_name == 'multi-task-gp':
            return MultiTaskGP(self.data, kernel=self.kernel, likelihood=self.likelihood, num_outputs=self.num_outputs)
        if model_name == 'VGP':
            return gpf.models.VGP(self.data, kernel=self.kernel, likelihood=self.likelihood,
                                  num_latent_gps=1)
        if model_name == 'GPR':
            return gpf.models.GPR(self.data, kernel=self.kernel)
        if model_name == 'GPMC':
            return gpf.models.GPMC(self.data, kernel=self.kernel, likelihood=self.likelihood, num_latent_gps=1)
        if model_name == 'difference':
            return DiffModel(self.data, kernel=self.kernel, likelihood=self.likelihood, num_outputs=self.num_outputs)
        raise Exception(f"Please enter a valid models name: one of {self.VALID_MODEL_NAMES}")

    def get_optimizer(self, optimizer_name):
        if optimizer_name == 'scipy':
            return gpf.optimizers.Scipy()
        if optimizer_name == 'adam':
            return tf.optimizers.Adam()
        raise Exception(f"Please enter a valid optimizer name: one of {self.VALID_OPTIMIZERS}")

    def train_model(self, disp=False):
        if isinstance(self.optimizer, gpf.optimizers.Scipy):
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for information
            self.optimizer.minimize(self.model.training_loss,
                                    variables=self.model.trainable_variables,
                                    options=dict(disp=disp),
                                    method="L-BFGS-B")
            tf.print("Done training models!")
        else:
            tf_optimization_step = tf.function(self.optimization_step)
            for epoch in range(self.MAXITER):
                loss = tf_optimization_step()
                epoch_id = epoch + 1
                if epoch_id % (self.MAXITER // 100) == 0:
                    tf.print(f"Epoch {epoch_id}: Loss (train) {loss}")
            tf.print("Done training models!")

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
