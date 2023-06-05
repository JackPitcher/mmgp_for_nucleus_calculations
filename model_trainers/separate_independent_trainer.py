import gpflow as gpf
from model_trainers.trainer import Trainer


class SeparateIndependentTrainer(Trainer):

    MODEL_NAME = 'SeparateIndependent'

    def __init__(self, Xs, Ys, inducing_size, optimizer=gpf.optimizers.Scipy()):
        super().__init__(Xs, Ys, inducing_size, optimizer=optimizer)

    def get_kernel(self, input_kernel):
        kern_list = [
            input_kernel for _ in range(self.num_outputs)
        ]
        kernel = gpf.kernels.SeparateIndependent(
            kern_list
        )
        return kernel
