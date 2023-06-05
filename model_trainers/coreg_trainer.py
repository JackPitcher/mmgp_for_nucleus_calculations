import gpflow as gpf
import numpy as np
from model_trainers.trainer import Trainer


class CoregTrainer(Trainer):

    MODEL_NAME = 'Coregion'

    def __init__(self, Xs, Ys, inducing_size, optimizer=gpf.optimizers.Scipy()):
        super().__init__(Xs, Ys, inducing_size, optimizer=optimizer)

    def get_kernel(self, input_kernel):
        kern_list = [
            input_kernel for _ in range(self.num_outputs)
        ]
        kernel = gpf.kernels.LinearCoregionalization(
            kern_list, W=np.random.randn(self.num_outputs, self.num_outputs)
        )
        return kernel
