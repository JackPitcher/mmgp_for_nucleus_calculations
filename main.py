import numpy as np
import matplotlib.pyplot as plt
from models.model_trainer import ModelTrainer
from models.multi_fidelity_deep_gp import MultiFidelityDeepGPTrainer as DeepTrainer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os

df = pd.read_csv('34Samples_dataset.csv')

seed = 1
num_hf_data = 12
np.random.seed(seed)
indices = np.random.choice(range(df.shape[0]), size=num_hf_data, replace=False)
test_indices = [i for i in range(df.shape[0]) if i not in indices]

x = df.iloc[:, 1:18]
y = df.iloc[:, 18:24]

x_train = x.iloc[indices, :]
y_train = y.iloc[indices, :]
x_test = x.iloc[test_indices, :]
y_test = y.iloc[test_indices, :]

X_train = np.vstack((
        np.hstack((x, np.zeros((x.shape[0], 1)), np.zeros((x.shape[0], 1)))),
        np.hstack((x, np.ones((x.shape[0], 1)), np.zeros((x.shape[0], 1)))),
        np.hstack((x, np.ones((x.shape[0], 1)) * 2, np.zeros((x.shape[0], 1)))),
        np.hstack((x_train, np.zeros((x_train.shape[0], 1)), np.ones((x_train.shape[0], 1)))),
        np.hstack((x_train, np.ones((x_train.shape[0], 1)), np.ones((x_train.shape[0], 1)))),
        np.hstack((x_train, np.ones((x_train.shape[0], 1)) * 2, np.ones((x_train.shape[0], 1))))
))

X_test = np.vstack((
        np.hstack((x_test, np.zeros((x_test.shape[0], 1)))),
        np.hstack((x_test, np.ones((x_test.shape[0], 1)))),
        np.hstack((x_test, np.ones((x_test.shape[0], 1)) * 2))
))

Y_train = np.vstack((
        np.hstack((np.reshape(np.array(y["GT_HF"]), (y.shape[0], 1)), np.zeros((y.shape[0], 1)), np.zeros((y.shape[0], 1)))),
        np.hstack((np.reshape(np.array(y["F_HF"]), (y.shape[0], 1)), np.ones((y.shape[0], 1)), np.zeros((y.shape[0], 1)))),
        np.hstack((np.reshape(np.array(y["CT_HF"]), (y.shape[0], 1)), np.ones((y.shape[0], 1)) * 2, np.zeros((y.shape[0], 1)))),
        np.hstack((np.reshape(np.array(y_train["GT"]), (y_train.shape[0], 1)), np.zeros((y_train.shape[0], 1)), np.ones((y_train.shape[0], 1)))),
        np.hstack((np.reshape(np.array(y_train["F"]), (y_train.shape[0], 1)), np.ones((y_train.shape[0], 1)), np.ones((y_train.shape[0], 1)))),
        np.hstack((np.reshape(np.array(y_train["CT"]), (y_train.shape[0], 1)), np.ones((y_train.shape[0], 1)) * 2, np.ones((y_train.shape[0], 1))))
))

Y_test = np.vstack((
        np.hstack((np.reshape(np.array(y_test["GT"]), (y_test.shape[0], 1)), np.zeros((y_test.shape[0], 1)))),
        np.hstack((np.reshape(np.array(y_test["F"]), (y_test.shape[0], 1)), np.ones((y_test.shape[0], 1)))),
        np.hstack((np.reshape(np.array(y_test["CT"]), (y_test.shape[0], 1)), np.ones((y_test.shape[0], 1)) * 2))
))

#Perform PCA to reduce dimensionality of the parameter space
scaler = StandardScaler()
scaler.fit(X_train[:, :-2])
x_train_data = scaler.transform(X_train[:, :-2])
x_test_data = scaler.transform(X_test[:, :-1])

n_dims = 7

if n_dims < x_train_data.shape[1]:
    pca = PCA(n_components=n_dims)
    pca.fit(x_train_data)
    x_train_data = pca.transform(x_train_data)
    x_test_data = pca.transform(x_test_data)
else:
    n_dims = x_train_data.shape[1]

X_train = np.hstack((x_train_data, X_train[:, -2:]))
X_test = np.hstack((x_test_data, X_test[:, -1:]))

model_name = 'DGP'
base_model = 'VGP'
base_kernels = ['ArcCosine', 'ArcCosine', 'Linear', 'RBF']
likelihood_name = 'Gaussian'

deep_trainer = DeepTrainer(
    data=(X_train, Y_train),
    optimizer_name='scipy',
    num_outputs=3
)

deep_trainer.construct_model(
    model_names=[base_model, base_model],
    base_kernels=base_kernels,
    likelihood_name=likelihood_name
)

deep_trainer.train_deep_model()


def plot_real_data(trainer, X_test):
    mean, var = trainer.predict(X_test, fidelity=None)
    mean_reshaped = np.reshape(mean, (mean.shape[0] // 3, 3), "F")
    var_reshaped = np.reshape(var, (var.shape[0] // 3, 3), "F")

    tasks = ["GT", "F", "CT"]
    ntasks = len(tasks)
    f, axes = plt.subplots(1, ntasks, figsize=(18, 4))
    xplot = np.arange(x.shape[0])

    for i in range(len(axes)):
        # Computed points as blue dots
        axes[i].scatter(xplot[test_indices], y_test[tasks[i]], c='b')
        # Plot training data as red dots
        axes[i].scatter(xplot[indices], y_train[tasks[i]], c='r')
        # Shade in confidence
        axes[i].errorbar(xplot[test_indices], mean_reshaped[:, i], yerr=2 * np.sqrt(np.abs(var_reshaped[:, i])), linestyle='none')
        axes[i].set_ylabel(tasks[i], size=20)
        axes[i].set_xlabel("Samples", size=20)
    path = f"images/{model_name}/{base_model}/{base_kernels[0]}"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(f"{path}/real_{n_dims}_{num_hf_data}_{seed}_{0}")


plot_real_data(deep_trainer, X_test)


