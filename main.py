from model_trainers.multi_fidelity_deep_gp import MultiFidelityDeepGPTrainer as DeepTrainer
from preprocessing.preprocess_imsrg_data import IMSRGPreprocessor
from visualization.plotter import Plotter

import os
import pandas as pd


def preprocess_data(df, split_string):
    final_df = pd.DataFrame(columns=['sample', 'E_parent', 'E_daughter', 'GT', 'F', 'T'])
    ret_df = pd.DataFrame(columns=['sample', 'E_parent', 'E_daughter', 'GT', 'F', 'T'])
    for name, group in df.groupby(['Energy bra']):
        operators = {}
        for i, row in group.iterrows():
            op_file_split = row['op_file'].split(split_string)
            sample = op_file_split[0].split('_')[-1]
            operator = op_file_split[1].split('_')[0]
            operators[operator] = float(row['Two'])
            e_parent = float(row['Energy ket'])
        ret_df[list(operators.keys())] = pd.DataFrame(operators, index=[0])
        ret_df['sample'] = int(sample)
        ret_df['E_parent'] = float(e_parent)
        ret_df['E_daughter'] = float(name)
        final_df = final_df.append(ret_df)
    final_df = final_df.set_index('sample').sort_index(ascending=True)
    return final_df


file_path = "data/new_samples_emax_10_m0nu.csv"
num_train_data = 20  # the number of data to train the highest level on
max_fidelity = 6  # modify this to add more fidelity layers. Note that it's one based
num_outputs = 4  # must change this if you change the number of tasks!
num_x_cols = 17  # how many x columns there are in the dataframe
num_pca_dims = None # how many pca dims to compress to; leave None for full resolution
# tasks is a dictionary: the key is the fidelity (higher number => higher fidelity)
#                        and the value is a list of columns associated with that fidelity
tasks = {
    "5": ["M0nu", "GT", "F", "T"],
    "4": ["M0nu_EMAX_10", "GT_EMAX_10", "F_EMAX_10", "T_EMAX_10"],
    "3": ["M0nu_EMAX_8", "GT_EMAX_8", "F_EMAX_8", "T_EMAX_8"],
    "2": ["M0nu_EMAX_6", "GT_EMAX_6", "F_EMAX_6", "T_EMAX_6"],
    "1": ["M0nu_EMAX_4", "GT_EMAX_4", "F_EMAX_4", "T_EMAX_4"],
    # "1": ["M0nu", "GT", "F", "T"],
    "0": ["M0nu_HF", "GT_HF", "F_HF", "T_HF"]
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
    model_names=[base_model for i in range(max_fidelity)],
    base_kernels=base_kernels,
    likelihood_name=likelihood_name
)

deep_trainer.train_deep_model()

y_train_df = preprocessor.get_y_data_as_df()
hf_tasks = tasks[str(max_fidelity - 1)]

plotter = Plotter(
    trainer=deep_trainer,
    X_test=X_test,
    Y_test=Y_test,
    Y_train=y_train_df,
    tasks=hf_tasks
)

train_inds, test_inds = preprocessor.get_indices()
path = f"images/new_samples/{model_name}/{base_model}/{base_kernels[0]}"
if not os.path.isdir(path):
    os.makedirs(path)

plotter.plot(
    num_data_points=y_train_df.shape[0] + Y_test.shape[0],
    train_indices=train_inds,
    test_indices=test_inds,
    path_to_save=f"{path}/fidelity_{max_fidelity}_dims_{num_pca_dims}_data_{num_train_data}_seed_{seed}_tasks_{hf_tasks}"
)

plotter.plot_prediction_vs_imsrg_data(
    ntasks=4,
    path_to_save=f"{path}/imsrg_fidelity_{max_fidelity}_dims_{num_pca_dims}_data_{num_train_data}_seed_{seed}_tasks_{hf_tasks}"
)
