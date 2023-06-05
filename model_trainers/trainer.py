import gpflow as gpf
import numpy as np
import tensorflow as tf
import tqdm as tqdm

from gpflow.config import (
    default_float,
    set_default_float
)
from gpflow.ci_utils import reduce_in_tests

set_default_float(np.float64)


class Trainer:
    MAXITER = reduce_in_tests(100000)
    KERNEL_TYPE = None
    MODEL_NAME = ""

    def __init__(self, Xs, Ys, inducing_size, optimizer=gpf.optimizers.Scipy()):
        """
        :param Xs: Inputs to the model, with one for each fidelity.
         Each X Expected to be a Tensor or Numpy array, with shape N_F x D, where N_F is the number of points at a
         particular fidelity, and D is the number of features.
        :param Ys: Outputs of the model, with one for each fidelity.
         Each Y expected to be a Tensor or Numpy array, with shape N_F x O, where N_F is the number of points at a
         particular fidelity, and O is the number of outputs.
        :param inducing_size: How many inducing points to use for the Sparse Variational GP model.
        :param optimizer: Which optimizer to use, e.g. Scipy or Adam optimizers.
        """

        self.Xs = Xs.copy()
        self.Ys = Ys.copy()
        self.max_fidelity = len(Ys)

        self.num_features = self.Xs[0].shape[1]
        self.input_dims = list(range(self.num_features))
        self.length_scales = tf.convert_to_tensor([1.0] * self.num_features, dtype=default_float())

        self.num_outputs = self.Ys[0].shape[1]
        self.output_dims = list(range(self.num_features, self.num_outputs + self.num_features))

        min_range = np.min([np.min(X) for X in Xs])
        max_range = np.max([np.max(X) for X in Xs])

        self.Z = np.tile(np.linspace(min_range, max_range, inducing_size)[:, None], (1, self.num_features))
        self.Z_with_mean = np.tile(np.linspace(0, 1, inducing_size)[:, None], (1, self.num_features + self.num_outputs))

        self.models = []
        self.optimizer = optimizer
        self.train_elbo = -np.inf
        self.test_elbo = -np.inf

        self.checkpoints = [None for i in range(self.max_fidelity)]

    def get_kernel(self, input_kernel):
        raise Exception("Trainer is an abstract class. Please instantiate one of its child classes.")

    def construct_model(self):
        kernel = self.get_kernel(
            gpf.kernels.SquaredExponential(lengthscales=self.length_scales)
        )
        inducing_variables = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(self.Z.copy())
        )
        self.models.append(
            gpf.models.SVGP(
                kernel,
                gpf.likelihoods.Gaussian(),
                inducing_variable=inducing_variables,
                num_latent_gps=self.num_outputs
            )
        )
        for fidelity in range(1, self.max_fidelity):
            kernel = self.get_kernel(
                gpf.kernels.SquaredExponential(lengthscales=self.length_scales, active_dims=self.input_dims)
                * (gpf.kernels.SquaredExponential(active_dims=self.output_dims)
                   + gpf.kernels.Linear(active_dims=self.output_dims))
                + gpf.kernels.SquaredExponential(lengthscales=self.length_scales, active_dims=self.input_dims)
            )
            inducing_variables = gpf.inducing_variables.SharedIndependentInducingVariables(
                gpf.inducing_variables.InducingPoints(self.Z_with_mean)
            )
            self.models.append(gpf.models.SVGP(
                kernel,
                gpf.likelihoods.Gaussian(),
                inducing_variable=inducing_variables,
                num_latent_gps=self.num_outputs
            ))

    def train(self):
        print("Start training...")
        for i in tqdm.tqdm(range(self.max_fidelity)):
            X, Y = self.Xs[i], self.Ys[i]
            self.optimize_model(self.models[i], (X, Y), i)
            for j in range(i + 1, self.max_fidelity):
                mu, _ = self.models[i].predict_f(self.Xs[j])  # Add the mean predicted by the current best model
                self.Xs[j] = np.concatenate((self.Xs[j][:, :self.num_features], mu), axis=1)
        self.train_elbo = self.calc_elbo((self.Xs[self.max_fidelity - 1], self.Ys[self.max_fidelity - 1]),
                                         self.max_fidelity)
        print(f"Done training! Final ELBO (higher is better): {self.train_elbo:.3}")

    def optimize_model(self, model, data, model_identity):
        log_dir = f"{self.MODEL_NAME}_{model_identity}"
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=3)
        manager.save()

        checkpoint_task = gpf.monitor.ExecuteCallback(manager.save)
        task_group = gpf.monitor.MonitorTaskGroup(checkpoint_task, period=3)
        monitor = gpf.monitor.Monitor(task_group)

        self.checkpoints[model_identity] = manager

        self.optimizer.minimize(
            model.training_loss_closure(data),
            variables=model.trainable_variables,
            step_callback=monitor,
            method='l-bfgs-b',
            options={"disp": 0, "maxiter": self.MAXITER, "maxfun": 10000}
        )

    def predict(self, X_test, Y_test=None, fidelity=None):
        if fidelity is None or fidelity > self.max_fidelity:
            fidelity = self.max_fidelity
        if fidelity <= 0:
            fidelity = 1
        X = X_test
        for i in range(fidelity):
            mu, var = self.models[i].predict_f(X)
            X = np.concatenate((X_test, mu), axis=1)
        if Y_test:
            self.test_elbo = self.calc_elbo((X, Y_test), fidelity=fidelity)
        return mu, var

    def calc_elbo(self, data, fidelity):
        return self.models[fidelity - 1].elbo(data)

    def get_elbo(self, training=False):
        return self.train_elbo if training else self.test_elbo
