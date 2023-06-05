import numpy as np
import pandas as pd


class Preprocessor:

    def __init__(self, filename, inputs, tasks, fidelities, training_size, seed=1):
        """
        :param filename: the path to the file
        :param max_fidelity: the maximum fidelity to train at
        :param training_size: the size of the training matrix, as integer or fraction
        :param seed: random seed
        """
        np.random.seed(seed)
        X, Y = self.read_data(filename, tasks, fidelities, inputs)
        self.max_fidelity = len(fidelities)
        if type(training_size) is int:
            self.X_train = X.sample(n=training_size)
        else:
            self.X_train = X.sample(frac=training_size)
        self.X_test = X[~X.index.isin(self.X_train.index)]
        self.Y_train = Y[X.index.isin(self.X_train.index)]
        self.Y_test = Y[~X.index.isin(self.X_train.index)]
        self.Y_mean = self.Y_train.mean()
        self.Y_std = self.Y_train.std()
        self.X_mean = self.X_train.mean()
        self.X_std = self.X_train.std()

    def read_data(self, filename, tasks, fidelities, inputs):
        outputs = [f"{task}{fid}" for task in tasks for fid in fidelities]
        df = pd.read_csv(filename)
        df = df[inputs + outputs]
        df = df.dropna(axis=0)
        df = df.reset_index(drop=True)
        return df[inputs], df[outputs]

    def scale_data(self):
        X_train_scaled = (self.X_train - self.X_mean) / self.X_std
        X_test_scaled = (self.X_test - self.X_mean) / self.X_std
        Y_train_scaled = (self.Y_train - self.Y_mean) / self.Y_std
        Y_test_scaled = (self.Y_test - self.Y_mean) / self.Y_std
        return X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled

    def unscale_y(self, df):
        return df * self.Y_std + self.Y_mean

    def unscale_x(self, df):
        return df * self.X_std + self.X_mean

    def unscale_y_task(self, array, task):
        return array * self.Y_std[task] + self.Y_mean[task]


