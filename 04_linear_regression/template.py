# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    N, D = features.shape
    M = mu.shape[0]
    fi = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            fi[i, j] = multivariate_normal.pdf(features[i, :], mean=mu[j, :], cov=sigma * np.identity(D))

    return fi
    

def _plot_mvn():
    X, t = load_regression_iris()
    N, D = X.shape
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
        
    for i in range(M):
        plt.plot(fi[:, i], label=f"Basis function {i}")
    
    plt.title("Output of Each Basis Function")
    plt.xlabel("Feature Value")
    plt.ylabel("Basis Function Output")
    plt.savefig("figures/plot_1_2_1.png", dpi=300)
    # plt.legend(loc="upper right)
    plt.show()



def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    # (fi.T * fi)
    fit_fi = np.dot(fi.T, fi)
    # (lambda * Identity Matrix)
    reg_fit_fi = fit_fi + lamda * np.identity(fi.shape[1])
    # Inverse Regularized fi transpose fi
    inverse_term = np.linalg.inv(reg_fit_fi)
    # (fi.T * targets)
    fit_targets = np.dot(fi.T, targets)
    # Maximum likelihood weights
    ml_weights = np.dot(inverse_term, fit_targets)
    
    return ml_weights

def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    fi = mvn_basis(features, mu, sigma)
    predictions = fi @ w
    return predictions


if __name__ == "__main__":
    # 1.1
    X, t = load_regression_iris()
    N, D = X.shape
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
    print(fi)

    # 1.2
    #_plot_mvn()

    # 1.3
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)
    print(wml)

    # 1.4
    prediction = linear_model(X, mu, sigma, wml)
    print(prediction)

    # 1.5
    print('Mean Square Error: ', np.mean((t - prediction)**2).round(4))
    plt.plot(t)
    plt.plot(prediction)
    plt.xlabel('Data Points')
    plt.ylabel('Target Values')
    plt.title('Target vs Predicted')
    plt.savefig('figures/1_5_1.png', dpi=300)
    plt.show()
    plt.plot(t - prediction)
    plt.xlabel('Data Points')
    plt.ylabel('Target Values')
    plt.title('Difference between Target and Predicted')
    plt.savefig('figures/1_5_2.png', dpi=300)
    plt.show()
