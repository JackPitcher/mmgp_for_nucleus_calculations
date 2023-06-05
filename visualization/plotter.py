import matplotlib.pyplot as plt
import numpy as np

from preprocessing.preprocess_imsrg_data import IMSRGPreprocessor


class Plotter:

    def __init__(self, trainer, preprocessor, X_test, Y_test, Y_train, tasks):
        self.trainer = trainer
        self.X_test = X_test
        self.Y_test = Y_test
        self.Y_train = Y_train
        self.tasks = tasks
        self.preprocessor = preprocessor
        self.y_mean = preprocessor.y_mean
        self.y_var = preprocessor.y_var
        self.Y_train = self.Y_train * self.y_var + self.y_mean
        self.Y_test = self.Y_test * self.y_var + self.y_mean

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

    def plot_prediction_vs_imsrg_data(self, path_to_save, min_size, max_size):
        ntasks = len(self.tasks)
        mean, var = self.trainer.predict(self.X_test)
        mean = np.reshape(mean, (mean.shape[0] // ntasks, ntasks), "F")
        var = np.reshape(var, (var.shape[0] // ntasks, ntasks), "F")
        nrows = ntasks // 2 if ntasks > 1 else 1
        f, axes = plt.subplots(nrows, ncols=ntasks // nrows, figsize=(18, 4))

        for i, ax in enumerate(axes.flatten()):
            task = self.tasks[i]
            mu = mean[:, i] * self.y_var[task] + self.y_mean[task]
            std = 2 * np.sqrt(var[:, i]) * self.y_var[task]

            ax.scatter(self.Y_train[task], self.Y_train[task], c='r', label="Training data", zorder=0)
            ax.errorbar(self.Y_test[task], mu, yerr=std, linestyle='none', marker='o', color='b',
                         label='Prediction', zorder=1)
            ax.plot([min_size - 1, max_size + 1], [min_size - 1, max_size + 1], c='k', linestyle=':')
            ax.set_ylabel(f'{task} (MM-DGP)')
            ax.set_xlabel(f'{task} (VS-IMSRG)')
            ax.set_xlim(min_size, max_size)
            ax.set_ylim(min_size, max_size)
            ax.legend(prop={'size': 4})

        plt.savefig(path_to_save)
        print(f"Plot saved to {path_to_save}")

