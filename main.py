from models.model_trainer import ModelTrainer
from models.multi_fidelity_deep_gp import MultiFidelityDeepGPTrainer as DeepTrainer
from preprocessing.preprocess_imsrg_data import IMSRGPreprocessor
from visualization.plotter import Plotter

import os

file_path = "data/34Samples_dataset.csv"
num_train_data = 12  # the number of data to train the highest level on
max_fidelity = 2  # modify this to add more fidelity layers. Note that it's one based
num_outputs = 4  # must change this if you change the number of tasks!
num_x_cols = 17  # how many x columns there are in the dataframe
num_pca_dims = None  # how many pca dims to compress to; leave None for full resolution
# tasks is a dictionary: the key is the fidelity (higher number => higher fidelity)
#                        and the value is a list of columns associated with that fidelity
tasks = {
        "1": ["GT", "F", "CT", "T"],
        "0": ["GT_HF", "F_HF", "CT_HF", "T_HF"]
    }
seed = 0
preprocessor = IMSRGPreprocessor(
    file_path=file_path,
    num_train_data=num_train_data,
    max_fidelity=max_fidelity,
    num_outputs=num_outputs,
    num_x_cols=num_x_cols,
    tasks=tasks,
    num_pca_dims=num_pca_dims,
    seed=seed
)

X_train, Y_train = preprocessor.get_training_data()
X_test, Y_test = preprocessor.get_testing_data()

model_name = 'DGP'
base_model = 'VGP'
base_kernels = ['ArcCosine', 'ArcCosine', 'Linear', 'RBF']
likelihood_name = 'Gaussian'

deep_trainer = DeepTrainer(
    data=(X_train, Y_train),
    optimizer_name='scipy',
    num_outputs=num_outputs
)

deep_trainer.construct_model(
    model_names=[base_model, base_model],
    base_kernels=base_kernels,
    likelihood_name=likelihood_name
)

deep_trainer.train_deep_model()

y_train_df = preprocessor.get_y_data_as_df()

plotter = Plotter(
    trainer=deep_trainer,
    X_test=X_test,
    Y_test=Y_test,
    Y_train=y_train_df,
    tasks=tasks[str(max_fidelity - 1)]
)

train_inds, test_inds = preprocessor.get_indices()
path = f"images/{model_name}/{base_model}/{base_kernels[0]}"
if not os.path.isdir(path):
    os.makedirs(path)

plotter.plot(
    num_data_points=y_train_df.shape[0] + Y_test.shape[0],
    train_indices=train_inds,
    test_indices=test_inds,
    path_to_save=f"{path}/real_{num_pca_dims}_{num_train_data}_{seed}_{num_outputs}_tasks"
)


