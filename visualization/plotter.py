import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    def __init__(self, trainer, X_test, Y_test, Y_train, tasks):
        self.trainer = trainer
        self.X_test = X_test
        self.Y_test = Y_test
        self.Y_train = Y_train
        self.tasks = tasks

    def plot(self, num_data_points, train_indices, test_indices, path_to_save):
        ntasks = len(self.tasks)
        mean, var = self.trainer.predict(self.X_test, fidelity=None)

        mean = np.reshape(mean, (mean.shape[0] // ntasks, ntasks), "F")
        var = np.reshape(2 * np.sqrt(var), (var.shape[0] // ntasks, ntasks), "F")

        nrows = ntasks // 2 if ntasks > 1 else 1
        f, axes = plt.subplots(nrows, ncols=ntasks // nrows, figsize=(18, 4))
        xplot = np.arange(num_data_points)

        for i, ax in enumerate(axes.flatten()):
            # Computed points as blue dots
            ax.scatter(xplot[test_indices], self.Y_test[self.tasks[i]], c='b')
            # Plot training data as red dots
            ax.scatter(xplot[train_indices], self.Y_train[self.tasks[i]], c='r')
            # Shade in confidence
            ax.errorbar(xplot[test_indices], mean[:, i], yerr=var[:, i], linestyle='none')
            ax.set_ylabel(self.tasks[i], size=20)
            ax.set_xlabel("Samples", size=20)
        plt.savefig(path_to_save)
        print(f"Plot saved to {path_to_save}")

    def plot_prediction_vs_imsrg_data(self, ntasks, path_to_save):
        mean, var = self.trainer.predict(self.X_test)
        mean = np.reshape(mean, (mean.shape[0] // ntasks, ntasks), "F")
        var = np.reshape(2 * np.sqrt(var), (var.shape[0] // ntasks, ntasks), "F")
        f, _ = plt.subplots(1, 1, figsize=(12, 10))
        plt.scatter(self.Y_train["M0nu"], self.Y_train["M0nu"], c='r', label="Training data", zorder=0)
        # plt.scatter(Y_test["M0nu"], mean[:, 0], c='b')
        plt.errorbar(self.Y_test["M0nu"], mean[:, 0], yerr=var[:, 0], linestyle='none', marker='o', color='b',
                     label='Prediction', zorder=1)
        plt.plot([-25, 16], [-25, 16], c='k', linestyle=':')
        plt.ylabel(r'$M^{0\nu}$ (MM-DGP)', size=20)
        plt.xlabel(r'$M^{0\nu}$ (VS-IMSRG)', size=20)
        plt.xlim(-25, 16)
        plt.ylim(-25, 16)
        plt.legend(prop={'size': 14})
        plt.savefig(path_to_save)
        print(f"Plot saved to {path_to_save}")

