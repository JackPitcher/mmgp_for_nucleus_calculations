"""Uses coregionalization to construct a multi-output Gaussian process for multiple tasks"""
import gpflow as gpf
import numpy as np


class MultiTaskGaussianProcess:

    MAXITER = 1000

    def __init__(self, output_dim, rank, optimizer='scipy'):
        base_kernel = gpf.kernels.Matern32(rank, active_dims=[0])
        coreg = gpf.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])
        coreg.W = np.random.rand(output_dim, rank)
        self.kernel = base_kernel * coreg
        self.likelihood = gpf.likelihoods.SwitchedLikelihood([gpf.likelihoods.Gaussian(),
                                                              gpf.likelihoods.Gaussian()])
        self.optimizer = optimizer
        self.model = None

    def create_model(self, x, y):
        model = gpf.models.VGP((x, y), kernel=self.kernel, likelihood=self.likelihood,
                               num_latent_gps=1)
        return model

    def get_optimizer(self):
        if self.optimizer == 'scipy':
            return gpf.optimizers.Scipy()
        return None

    def train(self, x, y):
        model = self.create_model(x, y)
        optimizer = self.get_optimizer()
        if optimizer is None:
            raise Exception("Unsupported optimizer: Please make sure the optimizer is valid.")
        optimizer.minimize(
            model.training_loss,
            variables=model.trainable_variables,
            options={"maxiter": self.MAXITER}
        )
        self.model = model

    def get_model(self):
        if self.model is None:
            raise Exception("Model uninitialized: Please train the model first.")
        return self.model
