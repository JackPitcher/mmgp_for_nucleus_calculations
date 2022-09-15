import gpflow as gpf
import numpy as np

from .multi_task_gaussian_process import MultiTaskGaussianProcess as MTGP


class ModelTrainer:

    MAXITER = 1000

    VALID_MODEL_NAMES = ['multi-task-gp']
    VALID_OPTIMIZERS = ['scipy']

    def __init__(self, model_name, optimizer_name, X, Y):
        self.optimizer = self.get_optimizer(optimizer_name)
        self.model = self.get_model(model_name, X, Y)

    def get_model(self, model_name, X, Y):
        if model_name == 'multi-task-gp':
            return MTGP(X, Y)
        raise Exception(f"Please enter a valid model name: one of {self.VALID_MODEL_NAMES}")

    def get_optimizer(self, optimizer_name):
        if optimizer_name == 'scipy':
            return gpf.optimizers.Scipy()
        raise Exception(f"Please enter a valid optimizer name: one of {self.VALID_OPTIMIZERS}")

    def train_model(self):
        self.optimizer.minimize(self.model.training_loss,
                                variables=self.model.trainable_variables,
                                options={"MAX_ITER": self.MAXITER})

    def predict(self, X_test):
        mean, var = self.model.predict_f(X_test)
        means = np.unique(mean.numpy())
        variances = np.unique(var, axis=0)
        variances = np.unique(variances, axis=1)
        return means, variances
