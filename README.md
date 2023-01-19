# Using Multi-fidelity, multi-output Gaussian Processes for Nucleus Calculations

## Overview

This project is an extension of the paper by A. Belley et al. [1] and is mostly maintained by Jack Pitcher under the supervision of Dr. Jason Holt at TRIUMF.
Our main purpose is to calculate nuclear matrix elements (NMEs) of double beta decay events.
Our project combines research into Multi-Fidelity Gaussian Processes [2] and Deep Gaussian Processes [3] to create the Multi-fidelity, multi-output deep Gaussian Process (MMDGP).
Most of this project is based off of [GPFlow][8]. 

The project is set up as follows:

* data contains data used by the project
* images contain images output by the project
* kernels contain different kinds of custom made kernels
* models contain both custom made models and model trainers
* notebooks contain Jupyter notebooks to test
* preprocessing contains the Preprocessor class, used to preprocess data
* visualization contains the Plotter class, used to plot graphs

The project uses data generated by [IMSRG][4] calculations of nuclear structure.

## Data

The data should be stored as a CSV file. It should include the x inputs (the LECs) as well as the outputs for any operator that you want to evaluate, for all fidelities that need to be computed.

## Preprocessing

The data is preprocessed by scaling using [Scikitlearn's Standard Scaler][5] and, optionally, [Principle Component Analysis][6].
The benefits of PCA depend on the data. It can help reduce overfitting by reducing the number of parameters and potentially reducing redundant parameters, but in practice it has had little effect thus far.
The data is formatted into Tensorflow Tensors to be used by the model. There are 4 tensors: X_test, X_train, Y_test, and Y_train.

* X_train is a Tensor of size (num_data * num_tasks * num_fidelities, num_features + 2), where num_features are the number of LECs and num_data is the number of training samples for each fidelity. The extra two columns are used to map the task (operator) and the fidelity that the particular sample is for. For example, suppose we want to calculate 2 operators, or tasks; call them 0 and 1. We want to do this for three different fidelities: 0, 1, and 2. Each sample would then have six entries in the tensor; one for each output, and for each of those, one for each fidelity.
* Y_train is a Tensor of size (num_data * num_tasks * num_fidelities, num_tasks * num_fidelities + 2). The 2 extra columns serve the exact same purpose as for the training data, and there is one column for each task at each fidelity.
* X_test is a Tensor of size (num_test_data * num_tasks, num_features + 1). Since testing data is generally only evaluated at the highest fidelity, since we only want to predict the highest quality data, it does not keep track of the fidelity like X_train. However, it still keeps track of the task of each testing sample.
* Y_test is a data frame and is only used for visualization purposes. 

## Model Trainers

Model trainers are classes that are used to train models in particular ways, rather than having to code everything in scripts. The Model Trainer is a generic trainer, while the Multi Fidelity Deep GP model trainer is for the MMDGP. They are made so that you only need to pass in the data and a few parameters and the model will train properly. The parameters required are:

* optimizer: Which optimizer to use. Currently only [the Scipy Optimizer][7] is implemented; however, any Tensorflow optimizer can theoretically be used.
* models: Which model to use for each level of fidelity. Generally the same model is used for all layers, but one could experiment with using different models at different levels. Names of implemented models are in the model trainer class, but any GPFlow model should work, or a model subclassed off of GPFlow.
* kernels: Which kernels to use for each model. The kernels are a list of 4 and are combined as follows:


    (kernel_0 * (kernel_1 + kernel_2 + kernel_3) + kernel_0) * coreg

    
Where coreg is the coregionalization kernel used for multi outputs. The particular combination of kernels is where MMDGP differs from regular DGP, which does not use a coregionalization kernel.
* Likelihoods: Which likelihood to use as the underlying assumption of the distribution of the model; Gaussian is standard.

## Models

Models contain custom models, subclassed off of GPFlow's base model. Currently, the custom models are works in progress and do not work.

## Kernels

Custom kernels can also be created, subclassed off of GPFlow's base kernel. Currently, custom kernels are works in progress and do not work.

## Visualization

The visualization package is for plotting results. There are currently two methods implemented: The first one creates a scatter plot of each task, with error bars, and shows the predicted value of the model; the second compares the predicted value with the value calculated by the IMSRG calculation.

## References

[1]: Belley, A., Payne, C. G., Stroberg, S. R., Miyagi, T., & Holt, J. D. (2021). Ab Initio Neutrinoless Double-Beta Decay Matrix Elements for Ca 48, Ge 76, and Se 82. Physical Review Letters, 126(4). https://doi.org/10.1103/PhysRevLett.126.042502

[2]: Brevault, L., Balesdent, M., & Hebbal, A. (2020). Overview of Gaussian process based multi-fidelity techniques with variable relationship between fidelities, application to aerospace systems. Aerospace Science and Technology, 107. https://doi.org/10.1016/j.ast.2020.106339

[3]: Jakkala, K. (2021). Deep Gaussian Processes: A Survey. http://arxiv.org/abs/2106.12135

[4]: https://github.com/ragnarstroberg/imsrg

[5]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

[6]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

[7]: https://docs.scipy.org/doc/scipy/reference/optimize.html

[8]: https://www.gpflow.org/