from typing import List, Dict, Union

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class IMSRGPreprocessor:
    """This class takes IMSRG data and preprocesses it into the appropriate format for the models"""
    file_path: str
    num_train_data: int
    max_fidelity: int
    num_outputs: int
    num_x_cols: int
    tasks: Dict[str, List[str]]
    seed: int = 0
    num_pca_dims: Union[int, None] = None

    def __post_init__(self):
        """
        Creates the training and testing datasets in the appropriate format
        :return: None
        """
        np.random.seed(self.seed)

        df = pd.read_csv(self.file_path)
        df = self.clean_df(df)

        indices = np.random.choice(range(df.shape[0]), size=self.num_train_data, replace=False)
        test_indices = [i for i in range(df.shape[0]) if i not in indices]
        self.train_indices = indices
        self.test_indices = test_indices

        x = df.iloc[:, 1:self.num_x_cols + 1]
        x_train = x.iloc[indices, :]
        x_test = x.iloc[test_indices, :]

        x, x_train, x_test = self.scale_data(x, x_train, x_test)
        if self.num_pca_dims is not None:
            x, x_train, x_test = self.perform_pca(x, x_train, x_test)

        y_cols = np.concatenate([value for value in self.tasks.values()])
        y = df[y_cols]
        y_train = y.iloc[indices, :]
        self.y_train_as_df = y_train

        self.X_train = np.vstack(
            [np.hstack((x, np.ones((x.shape[0], 1)) * i, np.ones((x.shape[0], 1)) * j))
             if j < self.max_fidelity - 1 else
             np.hstack((x_train, np.ones((x_train.shape[0], 1)) * i, np.ones((x_train.shape[0], 1)) * j))
             for j in range(self.max_fidelity) for i in range(self.num_outputs)]
        )

        self.Y_train = np.vstack([
            np.hstack(
                (np.reshape(np.array(y[self.tasks[str(j)][i]]), (y.shape[0], 1)), np.ones((y.shape[0], 1)) * i,
                 np.ones((y.shape[0], 1)) * j)) if j < self.max_fidelity - 1 else
            np.hstack((np.reshape(np.array(y_train[self.tasks[str(j)][i]]), (y_train.shape[0], 1)), np.ones((y_train.shape[0], 1)) * i,
                       np.ones((y_train.shape[0], 1)) * j))
            for j in range(self.max_fidelity) for i in range(self.num_outputs)
        ])

        self.X_test = np.vstack([
            np.hstack((x_test, np.ones((x_test.shape[0], 1)) * i)) for i in range(self.num_outputs)
        ])

        self.Y_test = y.iloc[test_indices, :]

    def perform_pca(self, x, x_train, x_test):
        """
        Perform PCA to reduce dimensionality of the parameter space
        :param x: The full dataset
        :param x_train: The training dataset
        :param x_test: The testing dataset
        :return: The scaled datasets
        """
        if self.num_pca_dims < x.shape[1]:
            pca = PCA(n_components=self.num_pca_dims)
            pca.fit(x_train)
            x = pca.transform(x)
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)

        return x, x_train, x_test

    def scale_data(self, x, x_train, x_test):
        """
        Scales the data
        :param x: The full dataset
        :param x_train: The training dataset
        :param x_test: The testing dataset
        :return: The scaled datasets
        """
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x = scaler.transform(x)
        return x, x_train, x_test

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataframe by dropping unused columns and removing rows with NaN values
        :param df: The dataframe to clean
        :return: The cleaned dataframe
        """
        x_cols = list(df.columns)[1:self.num_x_cols + 1]
        tasks = np.concatenate([value for value in self.tasks.values()])
        df = df[x_cols + list(tasks)]
        return df.dropna(axis=0)

    def get_training_data(self):
        return self.X_train, self.Y_train

    def get_testing_data(self):
        return self.X_test, self.Y_test

    def get_y_data_as_df(self):
        return self.y_train_as_df

    def get_indices(self):
        return self.train_indices, self.test_indices
