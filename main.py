import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from model.multi_task_gaussian_process import MultiTaskGaussianProcess
from model.difference_model import DifferenceModel
from model.model_trainer import ModelTrainer


def fh1(x):
    return np.power(6 * x - 2, 2) * np.sin(12 * x - 4) + 10


def fh2(x):
    return 1.5 * (x + 2.5) * np.sqrt(fh1(x))


def fh3(x):
    return 5 * x * x * np.sin(12 ** x)


def fl1(x):
    return 2 * (x + 2) * np.sqrt(fh1(x) - 1.3 * (np.power(6 * x - 2, 2) - 6 * x))


def fl2(x):
    return np.power(6 * x - 2, 2) * np.sin(8 * x - 4) + 10 - (np.power(6 * x - 2, 2) - 6 * x)


def fl3(x):
    return fh3(x) + (x * x * x * np.sin(3 * x - 0.5)) + 4 * np.cos(2 * x)


fh = [fh1, fh2, fh3]
fl = [fl1, fl2, fl3]

x_test = np.linspace(0, 1, 16)

sampler = qmc.LatinHypercube(d=1)
sample_hf = sampler.random(n=6).transpose()[0]
sample_lf = sampler.random(n=10).transpose()[0]
sample_lf = np.concatenate((sample_hf, sample_lf))

train_y_hf = np.stack([fh1(sample_hf), fh2(sample_hf), fh3(sample_hf)], -1)
train_y_lf = np.stack([fl1(sample_lf), fl2(sample_lf), fl3(sample_lf)], -1)

# sample_lf = tf.convert_to_tensor(np.transpose(np.vstack([sample_lf, sample_lf, sample_lf])))
# sample_hf = tf.convert_to_tensor(np.transpose(np.vstack([sample_hf, sample_hf, sample_hf])))
# train_y_hf = tf.convert_to_tensor(train_y_hf)
# train_y_lf = tf.convert_to_tensor(train_y_lf)

tf.config.run_functions_eagerly(True)
trainer = ModelTrainer(model_name="multi-task-gp", optimizer_name="scipy", X=sample_lf, Y=train_y_lf)
trainer.train_model()

mean, variance = trainer.predict(x_test)
print(mean, variance)

# diff = DifferenceModel(sample_hf, sample_lf, train_y_lf, train_y_hf)
# losses = diff.train()
# print(losses)

# mtgp = MultiTaskGaussianProcess(output_dim=6, rank=3)
# mtgp.train(sample_lf, train_y_lf)
# model = mtgp.get_model()
