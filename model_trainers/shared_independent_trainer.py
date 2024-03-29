import gpflow as gpf
from model_trainers.trainer import Trainer


class SharedIndependentTrainer(Trainer):

    MODEL_NAME = 'SharedIndependent'

    def __init__(self, Xs, Ys, inducing_size, optimizer=gpf.optimizers.Scipy()):
        super().__init__(Xs, Ys, inducing_size, optimizer=optimizer)

    def get_kernel(self, input_kernel):
        kernel = gpf.kernels.SharedIndependent(
            input_kernel,
            output_dim=self.num_outputs
        )
        return kernel
