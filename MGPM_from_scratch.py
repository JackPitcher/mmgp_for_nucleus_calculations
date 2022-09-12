from cmath import isnan
from operator import inv
from jax import grad
import jax.numpy as jnp
from numpy import argwhere
# import numpy as np
from numpy.random import default_rng

rng = default_rng()
import numpy.ma as ma
# import numpy
from jax.config import config

config.update("jax_debug_nans", True)
config.update('jax_enable_x64', True)
# import pyade.jade
from JADE import JADE


def is_pos_def(x):
    return jnp.all(jnp.linalg.eigvals(x) > 0)


class MGPT:
    """A Gaussian Process class for creating and exploiting
  a Gaussian Process model for multiple tasks"""

    def __init__(self, num_tasks, method='adam'):
        """Initialize a Gaussian Process model
    
    Input
    ------
    n_restarts: number of restarts of the local optimizer
    optimizer: algorithm of local optimization"""

        self.num_tasks = num_tasks
        self.method = method
        self.args = []

    def initialize_bounds(self):
        bounds = jnp.array(
            [[-20, 20]] * (self.n_features + 2 * self.num_tasks))  # Bounds on theta, sigma and v
        bounds = jnp.concatenate([bounds, jnp.array([[-10, 10]] * (
                    len(self.args) - (self.n_features + 2 * self.num_tasks)))])  # Bounds on A
        self.bounds = bounds

    def intialize_args(self, n_features):
        self.n_features = n_features
        lbt, ubt = -3, 2
        lbv, ubv = -1, 1
        lbs, ubs = -3, 2
        lba, uba = 0, 5
        # Generate starting points
        # self.sigma = sigma
        num_A = int(self.num_tasks * (self.num_tasks + 1) / 2)
        lhd_theta = jnp.zeros(self.n_features)
        lhd_sigma = jnp.zeros(self.num_tasks)
        lhd_v = jnp.zeros(self.num_tasks)
        lhd_A = jnp.ones(num_A)
        # Scale random samples to the given bounds
        initial_theta = (ubt - lbt) * lhd_theta + lbt
        initial_sig = (ubs - lbs) * lhd_sigma + lbs
        initial_A = (uba - lba) * lhd_A + lba
        initial_v = (ubv - lbv) * lhd_v + lbv
        self.args = jnp.concatenate((initial_theta, initial_sig, initial_v, initial_A))
        self.initialize_bounds()
        return self.args

    def get_args(self):
        p = self.args
        theta = p[:self.n_features]
        sig = p[self.n_features:self.n_features + self.num_tasks]
        v = p[self.n_features + self.num_tasks:self.n_features + 2 * self.num_tasks]
        A = p[self.n_features + 2 * self.num_tasks:]
        return theta, sig, v, A

    def set_args(self, theta, sig, v, A):
        args = []
        args.extend(theta)
        args.extend(sig)
        args.extend(v)
        args.extend(A)
        args = jnp.array(args)
        self.args = args

    def Corr_RBF(self, X1, X2, theta):
        """Construct the correlation matrix between X1 and X2
    
    Input
    -----
    X1, X2: 2D arrays, (n_samples, n_features)
    theta: array, correlation legnths for different dimensions
    
    Output
    ------
    K: the correlation matrix
    """
        K = jnp.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            try:
                # K[i,:] = np.exp(-np.sum(theta*(X1[i,:]-X2)**2, axis=1))
                K = K.at[i, :].set(jnp.exp(-jnp.sum(theta * (X1[i, :] - X2) ** 2, axis=1)))
            except IndexError:
                K = K.at[i, :].set(jnp.exp(-theta * (X1[i] - X2) ** 2))
                # K[i,:] = np.exp(-theta*(X1[i]-X2)**2)
        return K

    def Corr_outputs(self, A, v):
        """Construct the correlation matrix between output tasks using Choleksy decomposition
    
    Input
    -----
    A: 1D array of size w = Q(Q+1)/2 where Q is the number of tasks
    num_tasks: int, number of correlated outputs
    
    Output
    ------
    K: the correlation matrix
    """
        L = jnp.zeros([self.num_tasks, self.num_tasks])
        for i, ai in enumerate(A):
            k = int((-1 + jnp.sqrt(8 * i + 1)) / 2)
            l = int(i - k * (k + 1) / 2)
            L = L.at[k, l].set(ai)
            # L[k,l] = ai
        K = L @ L.transpose() + jnp.eye(self.num_tasks) * v
        return K

    def corr_Linear(self, sigma_l):
        """White Gaussian noise added to the kernel to represent uncertainty of inputs
    
    Input
    -----
    sigma_l: 1D array of the noise in the data
    
    Output
    ------
    K: the correlation matrix
    """
        K = sigma_l * jnp.eye(self.num_tasks)
        return K

    def K_mat(self, X1, X2, theta, A, v):
        return jnp.kron(self.Corr_RBF(X1, X2, theta), self.Corr_outputs(A, v))

    def Sigma_mat(self, sigma_l, size_x):
        return jnp.kron(self.corr_Linear(sigma_l), jnp.eye(size_x))

    def Neglikelihood(self, p):
        """Negative likelihood function
    
    Input
    -----
    theta: array, logarithm of the correlation legnths for different dimensions
    
    Output
    ------
    LnLike: likelihood value"""
        theta = p[:self.n_features]
        sigma_l = p[self.n_features:self.n_features + self.num_tasks]
        v = p[self.n_features + self.num_tasks:self.n_features + 2 * self.num_tasks]
        A = p[self.n_features + 2 * self.num_tasks:]
        v = 2 ** v
        theta = 2 ** theta  # To insure theta>0
        sigma_l = 2 ** sigma_l  # To insure sigam_l>0
        # Construct correlation matrix
        K = self.K_mat(self.X, self.X, theta, A, v)
        # print(np.linalg.eigvals(K))
        # print(is_pos_def(K))
        Sig = self.Sigma_mat(sigma_l, self.n_training_samples)
        K_Sig = K + Sig + jnp.eye(K.shape[0]) * 1e-8
        # print(np.linalg.eigvals(K_Sig))
        try:
            inv_K_Sig = jnp.linalg.inv(K_Sig) + jnp.eye(
                K.shape[0]) * 1e-8  # Inverse of correlation matrix
        except:
            print(p)
            print(jnp.linalg.eigvals(K_Sig))
            print(jnp.linalg.det(K))
            assert (1 == 0)

        # Compute log-likelihood
        logDetK = jnp.log(jnp.abs(jnp.linalg.det(K_Sig)))
        yKy = self.y.flatten().T @ inv_K_Sig @ self.y.flatten()
        LnLike = 0.5 * yKy + 0.5 * logDetK + 0.5 * self.n_training_samples * jnp.log(2 * jnp.pi)
        # Update attributes
        self.K_sig, self.inv_K_sig = K_Sig, inv_K_Sig
        self.p0 = p
        return LnLike

    def fit(self, X, y, **params):
        """GP model training
    
    Input
    -----
    X: 2D array of shape (n_samples, n_features)
    y: 2D array of shape (n_samples, 1)
    """
        # remove nan from data
        index_nan = jnp.argwhere(jnp.isnan(y).any(axis=1))
        if index_nan.size != 0:
            mask = jnp.ones(X.shape[0], dtype=bool)
            mask = mask.at[index_nan].set(False)
            y = y[mask, :]
            X = X[mask, :]
        self.n_training_samples = X.shape[0]
        self.X, self.y = X, y
        self.n_training_samples = X.shape[0]
        # Run the adma optimizer to find optimal parameters
        if self.method == 'adam':
            self.args, score = self.adam(self.args, **params)
        elif self.method == 'jade':
            self.args, score = self.jade(self.args, bounds=self.bounds, **params)

    def predict(self, X_test):
        """GP model predicting
    
    Input
    -----
    X_test: test set, array of shape (n_samples, n_features)
    
    Output
    ------
    f: GP predictions
    SSqr: Prediction variances"""
        theta, sigma, v, A = self.get_args()
        theta = 2 ** theta
        v = 2 ** v
        sigma = 2 ** sigma
        # Construct correlation matrix between test and train data
        k = self.K_mat(X_test, self.X, theta, A, v)
        k2 = k.T
        k_test_test = self.K_mat(X_test, X_test, theta, A, v)
        K = self.K_mat(self.X, self.X, theta, A, v)
        Sig = self.Sigma_mat(sigma, self.n_training_samples)
        # print(is_pos_def(Sig))
        K_Sig = K + Sig + jnp.eye(K.shape[0]) * 1e-8
        inv_K_Sig = jnp.linalg.inv(K_Sig)  # Inverse of correlation matrix
        # Mean prediction
        f = k @ inv_K_Sig @ self.y.flatten()
        # Variance prediction
        SSqr = k_test_test - k @ self.inv_K_sig @ k2
        if jnp.diag(SSqr).all() > 0: print("Cov ok")
        SSqr = jnp.sqrt(jnp.diag(jnp.abs(SSqr)))
        return f.reshape(X_test.shape[0], self.num_tasks), SSqr.reshape(X_test.shape[0],
                                                                        self.num_tasks)

    def adam(self, p0, n_iter=500, alpha=0.02, beta1=0.9, beta2=0.999, eps=1e-8, reinitialize=True):
        # generate an initial point
        self.p0 = p0
        score = self.Neglikelihood(self.p0)
        if (score > 1e6 or score < -1e6) and reinitialize == True:
            print(score)
            self.intialize_args(self.n_features)
            return self.adam(self.args, n_iter, alpha, beta1, beta2, eps, reinitialize=reinitialize)
        args = jnp.zeros([n_iter + 1, len(self.p0)])
        scores = jnp.zeros([n_iter + 1])
        ms = jnp.zeros([n_iter + 1, len(p0)])
        vs = jnp.zeros([n_iter + 1, len(p0)])
        mask = jnp.ones(n_iter + 1, dtype=bool)
        args = args.at[0, :].set(p0)
        scores = scores.at[0].set(score)
        # initialize first and second moments
        try:
            m = self.m
            v = self.v
        except:
            m = jnp.zeros(p0.size)
            v = jnp.zeros(p0.size)
        # run the gradient descent updates
        gradient = grad(self.Neglikelihood)
        for t in range(n_iter):
            # calculate gradient g(t)
            g = gradient(p0)
            # build a solution one variable at a time
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g ** 2
            mhat = m / (1.0 - beta1 ** (t + 1))
            vhat = v / (1.0 - beta2 ** (t + 1))
            p0 = p0 - alpha * mhat / (jnp.sqrt(vhat) + eps)
            # evaluate candidate point
            score = self.Neglikelihood(p0)
            if scores[t] - score > 1e2:
                mask = mask.at[t + 1].set(False)
            scores = scores.at[t + 1].set(score)
            args = args.at[t + 1, :].set(p0)
            ms = ms.at[t + 1, :].set(m)
            vs = vs.at[t + 1, :].set(v)
            # report progress
            print(f'{t} loss: {score}')
        index = jnp.argmin(scores[mask])
        score = scores[mask][index]
        self.NegLnlike = score
        self.m = ms[mask][index]
        self.v = vs[mask][index]
        p0 = args[mask, :][index, :]
        print(f"Best  optain score is {score}.")
        return [p0, score]

    def jade(self, p0, bounds=None, max_evals=1000, NP=10, c=0.1):
        nparams = len(p0)
        algorithm = JADE(nparams, bounds)
        p0, score = algorithm.run(self.Neglikelihood, NP=NP, max_evals=max_evals, c=c)
        return [p0, score]


class DifferenceModel(MGPT):

    def __init__(self, num_tasks, method='adam'):
        super().__init__(num_tasks, method=method)

    def initialize_bounds(self):
        bounds = jnp.array(
            [[-3, 10]] * (self.n_features + 2 * self.num_tasks))  # Bounds on theta, sigma and v
        bounds = jnp.concatenate([bounds, jnp.array([[-20, 20]] * (
                    len(self.args) - (self.n_features + 3 * self.num_tasks)))])  # Bounds on A
        bounds = jnp.concatenate([bounds, jnp.array([[-5, 5]] * self.num_tasks)])  # Bounds on rho
        self.bounds = bounds

    def intialize_args(self, n_features):
        self.n_features = n_features
        # lbt, ubt = -3,2
        # lbs, ubs = -3,2
        # lbv, ubv = -1,1
        # lba, uba = -5,5
        # lbp, ubp = -5,5
        lbt, ubt = -3, 10
        lbs, ubs = -3, 10
        lbv, ubv = -3, 10
        lba, uba = -20, 20
        lbp, ubp = -5, 5
        # self.sigma_l = sigma_l
        # self.sigma_h = sigma_h
        # Generate starting points
        num_A = int(self.num_tasks * (self.num_tasks + 1) / 2)
        lhd_theta = rng.standard_normal(self.n_features)
        lhd_sig = rng.standard_normal(self.num_tasks)
        lhd_A = rng.standard_normal(num_A)
        lhd_v = rng.standard_normal(self.num_tasks)
        lhd_rho = rng.standard_normal(self.num_tasks)
        # Scale random samples to the given bounds
        initial_theta = (ubt - lbt) * lhd_theta + lbt
        initial_sig = (ubs - lbs) * lhd_sig + lbs
        initial_v = (ubv - lbv) * lhd_v + lbv
        initial_A = (uba - lba) * lhd_A + lba
        initial_rho = (ubp - lbp) * lhd_rho + lbp
        self.args = jnp.concatenate((initial_theta, initial_sig, initial_v, initial_A, initial_rho))
        self.initialize_bounds()
        return self.args

    def get_args(self):
        p = self.args
        num_A = int(self.num_tasks * (self.num_tasks + 1) / 2)
        theta = p[: self.n_features]
        sig = p[self.n_features: self.n_features + self.num_tasks]
        v = p[self.n_features + self.num_tasks: self.n_features + 2 * self.num_tasks]
        A = p[self.n_features + 2 * self.num_tasks: self.n_features + 2 * self.num_tasks + num_A]
        rho = p[self.n_features + 2 * self.num_tasks + num_A:]
        return theta, sig, v, A, rho

    def set_args(self, theta, sig, v, A, rho):
        args = []
        args.extend(theta)
        args.extend(sig)
        args.extend(v)
        args.extend(A)
        args.extend(rho)
        args = jnp.array(args)
        self.args = args

    def K_mat(self, X1, X2, theta, A, v):
        return super().K_mat(X1, X2, theta, A, v)

    def Sigma_mat(self, sigma_l, size_x):
        return super().Sigma_mat(sigma_l, size_x)

    def Neglikelihood(self, p):
        """Negative likelihood function
    
    Input
    -----
    theta: array, logarithm of the correlation legnths for different dimensions
    
    Output
    ------
    LnLike: likelihood value"""
        num_A = int(self.num_tasks * (self.num_tasks + 1) / 2)
        theta = p[: self.n_features]
        sigma_h = p[self.n_features: self.n_features + self.num_tasks]
        v = p[self.n_features + self.num_tasks: self.n_features + 2 * self.num_tasks]
        A = p[self.n_features + 2 * self.num_tasks: self.n_features + 2 * self.num_tasks + num_A]
        rho = p[self.n_features + 2 * self.num_tasks + num_A:]

        self.p0 = p
        theta = 2 ** theta
        v = 2 ** v
        sigma_l = 2 ** self.sigma_l
        sigma_h = 2 ** sigma_h
        rho = sigma_h / sigma_l - 2 ** rho
        rho = jnp.tile(rho, self.n_training_samples)
        # Construct correlation matrix
        K = self.K_mat(self.X, self.X, theta, A, v)
        Sig = self.Sigma_mat(sigma_h, self.n_training_samples) - rho * self.Sigma_mat(sigma_l,
                                                                                      self.n_training_samples)
        # if is_pos_def(Sig) == False:
        #   Sig = np.zeros([self.num_tasks*self.n_training_samples,self.num_tasks*self.n_training_samples])
        K_Sig = K + Sig + jnp.eye(K.shape[0]) * 1e-5
        inv_K_Sig = jnp.linalg.inv(K_Sig) + jnp.eye(
            K.shape[0]) * 1e-5  # Inverse of correlation matrix
        # Compute log-likelihood
        logDetK = jnp.log(jnp.abs(jnp.linalg.det(K_Sig)))
        d = self.yh.flatten() - rho * self.yl.flatten()
        yKy = d.T @ inv_K_Sig @ d
        LnLike = 0.5 * yKy + 0.5 * logDetK + 0.5 * self.n_training_samples * jnp.log(2 * jnp.pi)
        # Update attributes
        self.K_sig, self.inv_K_sig = K_Sig, inv_K_Sig
        return LnLike

    def fit(self, X, yl, yh, sigma_l, **params):
        """GP model training
    
    Input
    -----
    X: 2D array of shape (n_samples, n_features)
    y: 2D array of shape (n_samples, 1)
    """
        index_nan_h = jnp.argwhere(jnp.isnan(yh).any(axis=1))
        index_nan_l = jnp.argwhere(jnp.isnan(yl).any(axis=1))
        if index_nan_h.size != 0 and index_nan_l.size != 0:
            index_nan = jnp.unique(jnp.concatenate(index_nan_h, index_nan_l))
            mask = jnp.ones(X.shape[0], dtype=bool)
            mask = mask.at[index_nan].set(False)
            yl = yl[mask, :]
            yh = yh[mask, :]
            X = X[mask, :]
        self.n_training_samples = X.shape[0]
        self.X, self.yl, self.yh = X, yl, yh
        self.n_training_samples = X.shape[0]
        self.sigma_l = sigma_l
        # Run the adma optimizer to find optimal parameters
        if self.method == 'adam':
            self.args, score = super().adam(self.args, **params)
        elif self.method == 'jade':
            self.args, score = super().jade(self.args, bounds=self.bounds, **params)

    def predict(self, X_test):
        """GP model predicting
    
    Input
    -----
    X_test: test set, array of shape (n_samples, n_features)
    
    Output
    ------
    f: GP predictions
    SSqr: Prediction variances"""
        theta, sigma_h, v, A, rho = self.get_args()
        theta = 2 ** theta
        v = 2 ** v
        sigma_l = 2 ** self.sigma_l
        sigma_h = 2 ** sigma_h
        rho = sigma_h / sigma_l - 2 ** rho
        rho = jnp.tile(rho, self.n_training_samples)
        # Construct correlation matrix between test and train data
        k = self.K_mat(X_test, self.X, theta, A, v)
        k2 = k.T
        k_test_test = self.K_mat(X_test, X_test, theta, A, v)
        K = self.K_mat(self.X, self.X, theta, A, v)
        Sig = self.Sigma_mat(sigma_h, self.n_training_samples) - rho * self.Sigma_mat(sigma_l,
                                                                                      self.n_training_samples)
        K_Sig = K + Sig
        inv_K_Sig = jnp.linalg.inv(K_Sig)  # Inverse of correlation matrix
        # Mean prediction
        d = self.yh.flatten() - rho * self.yl.flatten()
        f = k @ inv_K_Sig @ d.flatten()
        # Variance prediction
        SSqr = k_test_test - k @ self.inv_K_sig @ k2
        SSqr = jnp.sqrt(jnp.diag(jnp.abs(SSqr)))
        return f.reshape(X_test.shape[0], self.num_tasks), SSqr.reshape(X_test.shape[0],
                                                                        self.num_tasks)


class MMGP():

    def __init__(self, num_tasks, method='adam'):
        self.num_tasks = num_tasks
        self.args = []
        self.mgpt = MGPT(num_tasks, method=method)
        self.diff = DifferenceModel(num_tasks, method=method)

    def intialize_args(self, n_features):
        self.n_features = n_features
        self.mgpt.intialize_args(n_features)
        self.diff.intialize_args(n_features)

    def K_mat(self, Xl, Xh):
        theta_l, sig_l, v_l, A_l = self.mgpt.get_args()
        theta_d, sig_h, v_d, A_d, rho = self.diff.get_args()
        theta_l = 2 ** theta_l
        theta_d = 2 ** theta_d
        sig_l = 2 ** sig_l
        sig_h = 2 ** sig_h
        rho = sig_h / sig_l - 2 ** rho
        v_l = 2 ** v_l
        v_d = 2 ** v_d
        Kl_xlxl = self.mgpt.K_mat(Xl, Xl, theta_l, A_l, v_l)
        rho_l = jnp.tile(rho, Xl.shape[0])[jnp.newaxis].T
        rho = jnp.tile(rho, Xh.shape[0])[jnp.newaxis].T
        Kl_xhxl = rho * self.mgpt.K_mat(Xh, Xl, theta_l, A_l, v_l)
        Kl_xlxh = rho_l * self.mgpt.K_mat(Xl, Xh, theta_l, A_l, v_l)
        Kl_xhxh = rho * rho * self.mgpt.K_mat(Xh, Xh, theta_l, A_l, v_l)
        Kd_xhxh = self.diff.K_mat(Xh, Xh, theta_d, A_d, v_d)
        row1 = jnp.concatenate((Kl_xlxl, Kl_xlxh), axis=1)
        row2 = jnp.concatenate((Kl_xhxl, Kl_xhxh + Kd_xhxh), axis=1)
        return jnp.concatenate((row1, row2), axis=0)

    def K_mat_test(self, Xl, Xh, X_test):
        theta_l, sig_l, v_l, A_l = self.mgpt.get_args()
        theta_d, sig_h, v_d, A_d, rho = self.diff.get_args()
        theta_l = 2 ** theta_l
        theta_d = 2 ** theta_d
        sig_l = 2 ** sig_l
        sig_h = 2 ** sig_h
        rho = sig_h / sig_l - 2 ** rho
        v_l = 2 ** v_l
        v_d = 2 ** v_d
        rho = jnp.tile(rho, X_test.shape[0])[jnp.newaxis].T
        Kl_xlxh = rho * self.mgpt.K_mat(X_test, Xl, theta_l, A_l, v_l)
        # rho = np.tile(rho, Xh.shape[0])[np.newaxis].T
        Kl_xhxh = rho * rho * self.mgpt.K_mat(X_test, Xh, theta_l, A_l, v_l)
        Kd_xhxh = self.diff.K_mat(X_test, Xh, theta_d, A_d, v_d)
        return jnp.concatenate((Kl_xlxh, Kl_xhxh + Kd_xhxh), axis=1)

    def K_mat_test_T(self, Xl, Xh, X_test):
        theta_l, _, v_l, A_l = self.mgpt.get_args()
        theta_d, _, v_d, A_d, rho = self.diff.get_args()
        theta_l = 2 ** theta_l
        theta_d = 2 ** theta_d
        v_l = 2 ** v_l
        v_d = 2 ** v_d
        rho1 = jnp.tile(rho, Xl.shape[0])[jnp.newaxis].T
        Kl_xlxh = rho1 * self.mgpt.K_mat(Xl, X_test, theta_l, A_l, v_l)
        rho = jnp.tile(rho, Xh.shape[0])[jnp.newaxis].T
        Kl_xhxh = rho * rho * self.mgpt.K_mat(Xh, X_test, theta_l, A_l, v_l)
        Kd_xhxh = self.diff.K_mat(Xh, X_test, theta_d, A_d, v_d)
        return jnp.concatenate((Kl_xlxh, Kl_xhxh + Kd_xhxh), axis=0)

    def K_mat_test_test(self, X_test):
        theta_l, _, v_l, A_l = self.mgpt.get_args()
        theta_d, _, v_d, A_d, rho = self.diff.get_args()
        theta_l = 2 ** theta_l
        theta_d = 2 ** theta_d
        v_l = 2 ** v_l
        v_d = 2 ** v_d
        rho = jnp.tile(rho, X_test.shape[0])[jnp.newaxis].T
        Kl_xhxh = rho * rho * self.mgpt.K_mat(X_test, X_test, theta_l, A_l, v_l)
        Kd_xhxh = self.diff.K_mat(X_test, X_test, theta_d, A_d, v_d)
        return Kl_xhxh + Kd_xhxh

    def Sigma_mat(self, size_xl, size_xh):
        sigma_l = 2 ** self.diff.sigma_l
        sigma_h = 2 ** self.diff.get_args()[1]
        rho = self.diff.get_args()[-1]
        rho = sigma_h / sigma_l - 2 ** rho
        rho_l = jnp.tile(rho, size_xl)[jnp.newaxis].T
        rho = jnp.tile(rho, size_xh)[jnp.newaxis].T
        Sigl_xlxl = self.mgpt.Sigma_mat(sigma_l, size_xl)
        Sigl_xhxh = self.mgpt.Sigma_mat(sigma_l, size_xh)
        Sigh_xhxh = self.diff.Sigma_mat(sigma_h, size_xh) + rho * rho * Sigl_xhxh
        zeropad = jnp.zeros([size_xh, size_xl - size_xh])
        zeropad = jnp.kron(zeropad, jnp.eye(self.num_tasks))
        Sigl_xhxl = jnp.concatenate((zeropad, Sigl_xhxh), axis=1)
        Sigl_xhxl = rho * Sigl_xhxl
        zeropad = jnp.zeros([size_xl - size_xh, size_xh])
        zeropad = jnp.kron(zeropad, jnp.eye(self.num_tasks))
        Sigl_xlxh = jnp.concatenate((zeropad, Sigl_xhxh), axis=0)
        Sigl_xlxh = rho_l * Sigl_xlxh
        row1 = jnp.concatenate((Sigl_xlxl, Sigl_xlxh), axis=1)
        row2 = jnp.concatenate((Sigl_xhxl, Sigh_xhxh), axis=1)
        return jnp.concatenate((row1, row2), axis=0)

    def fit_lf(self, Xl, yl, **params):
        self.Xl, self.yl = Xl, yl
        print("Fitting low-fidelity model")
        self.mgpt.fit(Xl, yl, **params)
        print("Done fitting low-fidelity model.")

    def fit_diff(self, Xh, yh, **params):
        self.Xh = Xh
        yl_xh = self.yl[:self.Xh.shape[0], :]
        _, sigma_l, _, _ = self.mgpt.get_args()
        print("Fitting difference model.")
        self.diff.fit(Xh, yl_xh, yh, sigma_l, **params)
        index_nan_h = jnp.argwhere(jnp.isnan(yh).any(axis=1))
        if index_nan_h.size != 0:
            mask = jnp.ones(Xh.shape[0], dtype=bool)
            mask = mask.at[index_nan_h].set(False)
            yh = yh[mask, :]
        index_nan_l = jnp.argwhere(jnp.isnan(self.yl).any(axis=1))
        yl = self.yl
        if index_nan_l.size != 0:
            mask = jnp.ones(self.yl.shape[0], dtype=bool)
            mask = mask.at[index_nan_l].set(False)
            yl = self.yl[mask, :]
        self.y = jnp.concatenate((yl, yh), axis=0)
        print("Done fitting difference model.")

    def predict(self, X_test):
        Xl = self.mgpt.X
        Xh = self.diff.X
        K = self.K_mat(Xl, Xh)
        k = self.K_mat_test(Xl, Xh, X_test)
        k2 = self.K_mat_test_T(Xl, Xh, X_test)
        k_test_test = self.K_mat_test_test(X_test)
        Sig = self.Sigma_mat(Xl.shape[0], Xh.shape[0])
        K_Sig = K + Sig
        inv_K_Sig = jnp.linalg.inv(K_Sig)
        # Mean prediction
        f = k @ inv_K_Sig @ self.y.flatten()
        # # Variance prediction
        SSqr = k_test_test - k @ inv_K_Sig @ k2
        SSqr = jnp.sqrt(jnp.diag(jnp.abs(SSqr)))
        return f.reshape(X_test.shape[0], self.num_tasks), SSqr.reshape(X_test.shape[0],
                                                                        self.num_tasks)
